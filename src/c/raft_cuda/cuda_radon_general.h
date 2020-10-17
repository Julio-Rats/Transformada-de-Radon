#ifndef CUDA_RADON_GENERAL_H
#define CUDA_RADON_GENERAL_H

#include "siddon_iterator.h"
#include "cuda_utils.h"

namespace cuda {

   /// Computation of the Radon transform on the GPU.
   /**
    * This function computes the radon transform in the GPU
    * using the block-Radon cache-aware algorithm.
    *
    * The type image_t must provide the following public
    * typedefs:
    *
    * value_type and index_type;
    *
    * the following member data:
    *
    * index_type m, n : the numbers of lines and collumns of the image;
    * value_type tl_x, tl_y : Coordinates of the top-left vertex of image's boundary;
    * value_type br_x, br_y : Coordinates of the bottom-right vertex of image's boundary;
    *
    * and finally the member functions:
    *
    * value_t & image_t::operator()( index_t, index_t );
    * value_t const & image_t::operator()( index_t, index_t ) const;
    *
    * which are meant to provide references for the respective image pixels
    * using zero-based indexing.
    *
    * The result is the exact Radon transform, up to rounding errors, of an image
    * composed of square pixels of width ( image.br_x - image.tl_x ) / image.n
    * and height ( image.br_y - image.tl_y ) / image.m covering the square
    * [ image.tl_x, image.br_x ) x [ image.tl_y, image.br_y ).
    *
    * The samples are taken at the angles given, in radians, by:
    *
    * sino.
    *
    */
   template < int shm_blck_size, class image_t >
   __global__
   void radon_general(
      image_t const image,
      image_t sino,
      image_t angles
   )
   {
      typedef typename image_t::value_type value_t;
      typedef typename image_t::index_type index_t;

      // We wil keep a part of the image at shared memory:
      __shared__ value_t subimage[ shm_blck_size ][ shm_blck_size ];

      // Subimage interval:
      // TODO: temporarily use registers from the iterator
      // if compiler doesn't do it already.
      index_t i_0 = blockIdx.x * shm_blck_size;
      index_t i_1 = min( i_0 + shm_blck_size, image.m );
      index_t j_0 = blockIdx.y * shm_blck_size;
      index_t j_1 = min( j_0 + shm_blck_size, image.n );

      // Verify subimage is not empty:
      if ( ( i_0 >= i_1 ) || ( j_0 >= j_1 ) )
         return;

      // Create iterator for subimage:
      rtt::siddon_iterator< value_t, index_t > si;
      si.set_image(
         i_1 - i_0, j_1 - j_0,
         image.tl_x + j_0 * ( ( image.br_x - image.tl_x ) / image.n ),
         image.tl_y + i_0 * ( ( image.br_y - image.tl_y ) / image.m ),
         image.tl_x + j_1 * ( ( image.br_x - image.tl_x ) / image.n ),
         image.tl_y + i_1 * ( ( image.br_y - image.tl_y ) / image.m )
      );

      // Load subimages:
      // TODO: make sure global reads are coalesced (if not yet):
      for ( index_t j = j_0 + threadIdx.y; j < j_1; j += blockDim.y )
         for ( index_t i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
            subimage[ i - i_0 ][ j - j_0 ] = image( i, j );

      // Wait for copy to be done:
      __syncthreads();

      // Cycle through views:
      for ( index_t j = threadIdx.y; j < sino.n; j += blockDim.y )
      {
         // Set view in iterator:
         si.set_theta( angles( 0, j ) );

         // Compute intersecting ray-interval:
         // TODO: Again use registers from the iterator!
         // Top-left:
         value_t mx = ( si.tl_x_ * si.cos_theta_ ) + ( si.tl_y_ * si.sin_theta_ );
         value_t mn = mx;
         // Bottom-right:
         value_t tmp = ( si.br_x_ * si.cos_theta_ ) + ( si.br_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );
         // Bottom-left:
         tmp = ( si.tl_x_ * si.cos_theta_ ) + ( si.br_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );
         // Top-right:
         tmp = ( si.br_x_ * si.cos_theta_ ) + ( si.tl_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );

         // Compute intersecting ray-indices:
         i_0 = round( ( mn - sino.tl_y ) / ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );
         i_1 = round( ( mx - sino.tl_y ) / ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );

         if ( i_0 > i_1 )
         {
            index_t tmp = i_0;
            i_0 = i_1;
            i_1 = tmp;
         }

         i_0 = max( static_cast< index_t >( 0 ), i_0 );
         i_1 = min( sino.m, i_1 + static_cast< index_t >( 1 ) );

         // Cycle through rays:
         for ( index_t i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
         {
            // Set ray in iterator:
            si.set_t( sino.tl_y + i * ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );

            // Trace ray:
            value_t acc = static_cast< value_t >( 0.0 );
            while( si.valid() )
            {
               acc += ( subimage[ si.i() ][ si.j() ] * si.delta() );

               ++si;
            }

            // Store in global memory:
            atomicAdd( &( sino( i, j ) ), acc );
         }
      }
   }

   template < int shm_blck_size, class image_t >
   __global__
   void radon_transpose_general(
      image_t const sino,
      image_t image,
      image_t angles
   )
   {
      typedef typename image_t::value_type value_t;
      typedef typename image_t::index_type index_t;

      // We wil keep a part of the image at shared memory:
      __shared__ value_t subimage[ shm_blck_size ][ shm_blck_size ];

      // Subimage interval:
      // TODO: temporarily use registers from the iterator
      // if compiler doesn't do it already.
      index_t i_0 = blockIdx.x * shm_blck_size;
      index_t i_1 = min( i_0 + shm_blck_size, image.m );
      index_t j_0 = blockIdx.y * shm_blck_size;
      index_t j_1 = min( j_0 + shm_blck_size, image.n );

      // Verify subimage is not empty:
      if ( ( i_0 >= i_1 ) || ( j_0 >= j_1 ) )
         return;

      // Create iterator for subimage:
      rtt::siddon_iterator< value_t, index_t > si;
      si.set_image(
         i_1 - i_0, j_1 - j_0,
         image.tl_x + j_0 * ( ( image.br_x - image.tl_x ) / image.n ),
         image.tl_y + i_0 * ( ( image.br_y - image.tl_y ) / image.m ),
         image.tl_x + j_1 * ( ( image.br_x - image.tl_x ) / image.n ),
         image.tl_y + i_1 * ( ( image.br_y - image.tl_y ) / image.m )
      );

      // Load subimages:
      // TODO: make sure global reads are coalesced (if not yet):
      for ( index_t j = j_0 + threadIdx.y; j < j_1; j += blockDim.y )
         for ( index_t i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
            subimage[ i - i_0 ][ j - j_0 ] = image( i, j );

      // Wait for copy to be done:
      __syncthreads();

      // Cycle through views:
      for ( index_t j = threadIdx.y; j < sino.n; j += blockDim.y )
      {
         // Set view in iterator:
         si.set_theta( angles( 0, j ) );

         // Compute intersecting ray-interval:
         // TODO: Again use registers from the iterator!
         // Top-left:
         value_t mx = ( si.tl_x_ * si.cos_theta_ ) + ( si.tl_y_ * si.sin_theta_ );
         value_t mn = mx;
         // Bottom-right:
         value_t tmp = ( si.br_x_ * si.cos_theta_ ) + ( si.br_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );
         // Bottom-left:
         tmp = ( si.tl_x_ * si.cos_theta_ ) + ( si.br_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );
         // Top-right:
         tmp = ( si.br_x_ * si.cos_theta_ ) + ( si.tl_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );

         // Compute intersecting ray-indices:
         i_0 = round( ( mn - sino.tl_y ) / ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );
         i_1 = round( ( mx - sino.tl_y ) / ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );

         if ( i_0 > i_1 )
         {
            index_t tmp = i_0;
            i_0 = i_1;
            i_1 = tmp;
         }

         i_0 = max( static_cast< index_t >( 0 ), i_0 );
         i_1 = min( sino.m, i_1 + static_cast< index_t >( 1 ) );

         // Cycle through rays:
         for ( index_t i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
         {
            // Set ray in iterator:
            si.set_t( sino.tl_y + i * ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );

            // Trace ray:
            value_t data = sino( i, j );
            while( si.valid() )
            {
               atomicAdd( &( subimage[ si.i() ][ si.j() ] ), data * si.delta() );

               ++si;
            }
         }
      }

      // Waits for all views to be traced:
      __syncthreads();

      // Copy result to global memory:
      // TODO: make sure global writes are coalesced (if not yet)
      // and that there are no bank conflicts:
      i_0 = blockIdx.x * shm_blck_size;
      i_1 = min( i_0 + shm_blck_size, image.m );
      for ( int j = j_0 + threadIdx.y; j < j_1; j += blockDim.y )
         for ( int i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
            image( i, j ) = subimage[ i - i_0 ][ j - j_0 ];
   }

   template < class image_t, class op_t >
   __global__ void operate( image_t x, image_t const y, op_t op )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = op( x( i, j ), y( i, j ) );
   }

   template < class image_t >
   __global__ void sum( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) += y( i, j );
   }

   template < class image_t >
   __global__ void prod( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) *= y( i, j );
   }

   template < class image_t >
   __global__ void non_negative_projection( image_t x )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = x( i, j ) < static_cast< typename image_t::value_type >( 0.0 ) ?
               static_cast< typename image_t::value_type >( 0.0 ) :
               x( i, j );
   }

   template < class image_t >
   __global__ void lower_threshold( image_t x, typename image_t::value_type threshold )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = ( x( i, j ) < threshold ) ? threshold : x( i, j );
   }

   template < class image_t >
   __global__ void sub( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) -= y( i, j );
   }

   template < class image_t >
   __global__ void sub( image_t x, typename image_t::value_type alpha )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) -= alpha;
   }

   template < class image_t >
   __global__ void axpy( image_t x, image_t const y, typename image_t::value_type alpha )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) += alpha * y( i, j );
   }

   template < class image_t >
   __global__ void signal( image_t x )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = ( x( i, j ) >= 0.0 ) ? ( ( x( i, j ) > 0.0 ) ? 1.0 : 0.0 ) : -1.0;
   }

   template < class image_t >
   __global__ void set( image_t x, typename image_t::value_type alpha )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = alpha;
   }

   template < class image_t >
   __global__ void set( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = y( i, j );
   }

   template < class image_t >
   __global__ void divide( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) /= y( i, j );
   }

   template < class image_t >
   __global__ void ldivide( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = y( i, j ) / x( i, j );
   }

   template < class image_t >
   __global__ void multiply( image_t x, image_t const y )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) *= y( i, j );
   }

   template < class image_t >
   __global__ void multiply( image_t x, typename image_t::value_type alpha )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) *= alpha;
   }

   __device__ double exp_over( double x )
   {
      return( exp( x ) );
   }
   __device__ float exp_over( float x )
   {
      return( __expf( x ) );
   }

   __device__ double log_over( double x )
   {
      return( log( x ) );
   }
   __device__ float log_over( float x )
   {
      return( __logf( x ) );
   }

   template < class image_t >
   __global__ void exp( image_t x )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = exp_over( x( i, j ) );
   }

   template < class image_t >
   __global__ void neg_exp( image_t x )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = exp_over( -x( i, j ) );
   }

   template < class image_t >
   __global__ void log( image_t x )
   {
      typedef typename image_t::index_type index_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
            x( i, j ) = log_over( x( i, j ) );
   }

   template < class image_t >
   __global__ void tv_grad( image_t const x, image_t t )
   {
      typedef typename image_t::index_type index_t;
      typedef typename image_t::value_type value_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
         {
            t( i, j ) = static_cast< value_t >( 0.0 );

            value_t x_ij = x( i, j );

            // Neighbouring elements:
            index_t i_p = ( i + 1 ) % x.m;
            index_t i_m = i - 1;
            if ( i_m < 0 )
               i_m += x.m;
            index_t j_p = ( j + 1 ) % x.n;
            index_t j_m = j - 1;
            if ( j_m < 0 )
               j_m += x.n;

            value_t x_im = x( i_m, j );
            value_t x_jm = x( i, j_m );

            // Main component:
            value_t del_h = x_ij - x_jm;
            value_t del_v = x_ij - x_im;
            value_t del_vh = hypotf( del_h, del_v );
            if ( del_vh )
               t( i, j ) += ( del_h + del_v ) / del_vh;

            value_t x_ip = x( i_p, j );
            value_t x_jp = x( i, j_p );
            value_t x_imjp = x( i_m, j_p );

            // Secondary horizontal component:
            del_h = x_ij - x_jp;
            del_v = x_jp - x_imjp;
            del_vh = hypotf( del_h, del_v );
            if ( del_vh )
               t( i, j ) += del_h / del_vh;

            value_t x_ipjm = x( i_p, j_m );

            // Secondary vertical component:
            del_h = x_ip - x_ipjm;
            del_v = x_ij - x_ip;
            del_vh = hypotf( del_h, del_v );
            if ( del_vh )
               t( i, j ) += del_v / del_vh;
         }
   }

   template < class image_t >
   __global__ void tv_del( image_t const x, image_t t )
   {
      typedef typename image_t::index_type index_t;
      typedef typename image_t::value_type value_t;

      for ( index_t j = blockIdx.x; j < x.n; j += gridDim.x )
         for ( index_t i = threadIdx.x; i < x.m; i += blockDim.x )
         {
            // Neighbouring elements:
            index_t i_m = i - 1;
            if ( i_m < 0 )
               i_m += x.m;
            index_t j_m = j - 1;
            if ( j_m < 0 )
               j_m += x.n;

            value_t x_ij = x( i, j );
            t( i, j ) = hypotf( x_ij - x( i, j_m ), x_ij - x( i_m, j ) );
         }
   }

   template < class image_t >
   typename image_t::value_type host_sum( image_t & x )
   {
      typedef typename image_t::index_type index_t;
      typename image_t::value_type sum( 0.0 );
      for ( index_t i( 0 ); i < x.m; ++i )
         for ( index_t j( 0 ); j < x.n; ++j )
            sum += x( i, j );

      return( sum );
   }

   template < class image_t >
   typename image_t::value_type host_abs_sum( image_t & x )
   {
      typedef typename image_t::index_type index_t;
      typename image_t::value_type sum( 0.0 );
      for ( index_t i( 0 ); i < x.m; ++i )
         for ( index_t j( 0 ); j < x.n; ++j )
            sum += std::abs( x( i, j ) );

      return( sum );
   }

   template < class image_t >
   typename image_t::value_type host_sqr_sum( image_t & x )
   {
      typedef typename image_t::index_type index_t;
      typename image_t::value_type sum( 0.0 );
      for ( index_t i( 0 ); i < x.m; ++i )
         for ( index_t j( 0 ); j < x.n; ++j )
            sum += x( i, j ) * x( i, j );

      return( sum );
   }

} // namespace cuda

#endif // #ifndef CUDA_RADON_H
