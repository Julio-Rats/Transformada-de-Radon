#ifndef CUDA_IMAGE2_H
#define CUDA_IMAGE2_H

#include <cstdlib>
#include <new>
#include <fstream>
#include <string>

namespace cuda {

   /// Class template for 2-dimensional arrays;
   /**
    * This class template provides a C-style (row-major) 2-d array
    * which can be accessed from the host or the device.
   */
   template < class value_t = double, class index_t = int >
   class array2_t {

      public:

         typedef value_t value_type; ///< Type of elements.
         typedef index_t index_type; ///< Type of indices.

         value_t * device_data_pointer; ///< Pointer to device storage.
         value_t * host_data_pointer; ///< Pointer to host storage.
         index_t m; ///< Number of lines.
         index_t n; ///< Number of columns.

         /// Constructor.
         /**
          * Initialize array, but does not allocate any memory. The
          * functions device_init() and host_init() perform the allocation.
          *
          * \param m Number of lines.
          * \param n Number of columns.
          */
         inline __device__ __host__
         array2_t( index_t m = 0, index_t n = 0 )
         : device_data_pointer( 0 ), host_data_pointer( 0 ),
         m( m ), n( n )
         {}

         /// Element access.
         /**
          * \return A reference to the required element.
          *
          * \param i Zero-based line index.
          * \param j Zero-based column index.
          */
         inline __device__ __host__
         value_t & operator()( index_t i, index_t j )
         {
   #ifdef __CUDA_ARCH__
            return( *( device_data_pointer + ( i + ( j * m ) ) ) );
   #else
            return( *( host_data_pointer + ( i + ( j * m ) ) ) );
   #endif
         }

         /// Const element access.
         /**
          * \return A const reference to the required element.
          *
          * \param i Zero-based line index.
          * \param j Zero-based column index.
          */
         inline __device__ __host__
         value_t const & operator()( index_t i, index_t j ) const
         {
   #ifdef __CUDA_ARCH__
            return( *( device_data_pointer + ( i + ( j * m ) ) ) );
   #else
            return( *( host_data_pointer + ( i + ( j * m ) ) ) );
   #endif
         }

         /// Request memory on the host.
         /**
          * This function asks for memory on the host side.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void host_init( void )
         {
            if ( !host_data_pointer )
               host_data_pointer = static_cast< value_t * >( std::malloc( m * n * sizeof( value_t ) ) );
            if ( !host_data_pointer )
               throw std::bad_alloc();
         }

         /// Request memory on the device.
         /**
          * This function asks for memory on the device side.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void device_init( void )
         {
            if ( !device_data_pointer )
               if (
                  cudaMalloc(
                     reinterpret_cast< void ** >( &device_data_pointer ),
                     m * n * sizeof( value_t )
                  ) != cudaSuccess
               )
                  throw std::bad_alloc();
         }

         /// Request memory on the host and the device.
         /**
          * This function asks for memory on the host and the device.
          * Throws std::bad_allc if failed.
          */
         inline __host__
         void init( void )
         {
            host_init();
            device_init();
         }

         /// Copy data from host to device.
         /**
          * This function copies data from the host to the device.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void host2device( void )
         {
            if (
               cudaMemcpy(
                  device_data_pointer,
                  host_data_pointer,
                  m * n * sizeof( value_t ),
                  cudaMemcpyHostToDevice
               ) != cudaSuccess
            )
               throw std::bad_alloc(); // TODO: Throw more meaningfull exception
         }

         /// Copy data from device to host.
         /**
          * This function copies data from the device to the host.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void device2host( void )
         {
            if (
               cudaMemcpy(
                  host_data_pointer,
                  device_data_pointer,
                  m * n * sizeof( value_t ),
                  cudaMemcpyDeviceToHost
               ) != cudaSuccess
            )
               throw std::bad_alloc(); // TODO: Throw more meaningfull exception
         }

         /// Free host memory.
         /**
          * Relinquish memory to host.
          */
         inline __host__
         void host_destroy( void )
         {
            std::free( host_data_pointer );
            host_data_pointer = 0;
         }

         /// Free device memory.
         /**
          * Relinquish memory to device.
          */
         inline __host__
         void device_destroy( void )
         {
            cudaFree( device_data_pointer );
            device_data_pointer = 0;
         }

         /// Free memory.
         /**
          * Relinquish memory to both the host end the device.
          */
         inline __host__
         void destroy( void )
         {
            device_destroy();
            host_destroy();
         }
   };

   template< class value_t = double, class index_t = int >
   class image2_t : public array2_t< value_t, index_t > {

      public:

         using array2_t< value_t, index_t >::m;
         using array2_t< value_t, index_t >::n;
         using array2_t< value_t, index_t >::host_data_pointer;
         using array2_t< value_t, index_t >::device_data_pointer;
         using array2_t< value_t, index_t >::host_destroy;
         using array2_t< value_t, index_t >::host_init;

         typedef value_t value_type; ///< Type of elements.
         typedef index_t index_type; ///< Type of indices.

         value_t tl_x;
         value_t tl_y;
         value_t br_x;
         value_t br_y;

         /// Constructor.
         /**
          * Initialize image, but does not allocate any memory. The
          * functions device_init() and host_init() perform the allocation.
          *
          * \param m Number of lines;
          * \param n Number of columns;
          * \param tl_x x-coordinate of top-left corner of the image;
          * \param tl_y y-coordinate of top-left corner of the image;
          * \param br_x x-coordinate of bottom-right corner of the image;
          * \param br_y y-coordinate of bottom-right corner of the image;
          */
         inline __device__ __host__
         image2_t(
            index_t m = 0, index_t n = 0,
            value_t tl_x = -1.0, value_t tl_y =  1.0,
            value_t br_x =  1.0, value_t br_y = -1.0
         )
         : array2_t< value_t, index_t >( m, n ),
           tl_x( tl_x ), tl_y( tl_y ),
           br_x( br_x ), br_y( br_y )
         {}

         template < class stream_t >
         inline __host__
         void read( stream_t & istream )
         {
            // Read image dimensions:
            istream.read( reinterpret_cast< char * >( &m ), sizeof( m ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading number of lines failed!" );
            istream.read( reinterpret_cast< char * >( &n ), sizeof( n ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading number of columns failed!" );

            // Setup space for image:
            if ( host_data_pointer )
               host_destroy();

            host_init();

            // Read main data:
            istream.read( reinterpret_cast< char * >( host_data_pointer ), m * n * sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading image data failed!" );

            // Read image corners:
            istream.read( reinterpret_cast< char * >( &tl_x ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading top-left x-coord failed!" );
            istream.read( reinterpret_cast< char * >( &tl_y ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading top-left y-coord failed!" );
            istream.read( reinterpret_cast< char * >( &br_x ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading bottom-right x-coord failed!" );
            istream.read( reinterpret_cast< char * >( &br_y ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image2_t::read(): Reading bottom-right y-coord failed!" );
         }

         inline __host__
         void read( char const * fname )
         {
            std::ifstream ifs( fname, std::ifstream::binary );
            if ( !ifs.good() )
               throw std::ios_base::failure( "image2_t::read(): Failed to open file!" );

            read( ifs );
         }

         inline __host__
         void read( std::string const & fname )
         {
            std::ifstream ifs( fname.c_str(), std::ifstream::binary );
            if ( !ifs.good() )
               throw std::ios_base::failure( "image2_t::read(): Failed to open file!" );

            read( ifs );
         }

         template < class stream_t >
         inline __host__
         void write( stream_t & ostream ) const
         {
            // Write image dimensions:
            ostream.write( reinterpret_cast< char const * >( &m ), sizeof( m ) );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing number of lines failed!" );
            ostream.write( reinterpret_cast< char const * >( &n ), sizeof( n ) );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing number of column failed!" );

            // Write main data:
            ostream.write(
               reinterpret_cast< char const * >( host_data_pointer ),
               m * n * sizeof( value_t )
            );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing image data failed!" );

            // Write image corners:
            ostream.write( reinterpret_cast< char const * >( &tl_x ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing top-left x-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &tl_y ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing top-left y-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &br_x ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing bottom-right x-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &br_y ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image2_t::write(): Writing bottom-right y-coord failed!" );
         }

         inline __host__
         void write( char const * fname ) const
         {
            std::ofstream ofs( fname, std::ifstream::binary );
            if ( !ofs.good() )
               throw std::ios_base::failure( "image2_t::write(): Failed to open file!" );

            write( ofs );
         }

         inline __host__
         void write( std::string const & fname ) const
         {
            std::ofstream ofs( fname.c_str(), std::ifstream::binary );
            if ( !ofs.good() )
               throw std::ios_base::failure( "image2_t::write(): Failed to open file!" );

            write( ofs );
         }

         value_type h_sampling_distance( void ) const
         {
            return( ( br_x - tl_x ) / ( n - 1 ) );
         }

         value_type v_sampling_distance( void ) const
         {
            return( ( tl_y - br_y ) / ( m - 1 ) );
         }
   };
} // namespace cuda

#endif // #ifndef CUDA_ARRAY_H
