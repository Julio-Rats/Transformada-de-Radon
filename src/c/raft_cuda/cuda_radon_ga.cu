#include "cuda_radon_general.h"
#include "cuda_image2.h"
#include <vector>
#include <algorithm>

double PI   = 3.1415926535897932385;    ///< \f$\pi\f$.
double QPI  = 7.8539816339744830962e-1; ///< \f$\pi/4\f$.
double GA   = 1.9416110387254665773;    ///< Golden angle in radians: \f$2\pi/(1 + \sqrt 5)\f$.

template< class d_type >
std::vector< d_type > golden_angles( int n, d_type first_angle = 0.0 )
{
   // Vector for storing the angles:
   std::vector< d_type > retval;
   retval.reserve( n );

   // Compute sequence of angles:
   retval.push_back( first_angle );
   for( int i = 1; i < n; ++i )
      retval.push_back( retval[ i - 1 ] + GA );

   // Constrain, modulo \f$\pi\f$, to the interval \f$( \pi / 4, 5\pi / 4 ]\f$
   for( int i = 0; i < n; ++i )
   {
      d_type angle = retval[ i ] - QPI;
      int mul = angle / PI;
      angle -= PI * mul;
      angle = ( angle <= 0.0 ) ? angle + PI : angle;
      retval[ i ] = angle + QPI;
   }

   // Sort angles:
   std::sort( retval.begin(), retval.end() );

   return retval;
}

#define SYNC_AND_CHECK \
cudaDeviceSynchronize(); \
if ( cudaGetLastError() != cudaSuccess ) \
{ \
   printf( "Kernel launch failed above line %d: %s\n", __LINE__, cudaGetErrorString( cudaGetLastError() ) ); \
   std::exit( EXIT_FAILURE ); \
}

typedef cuda::image2_t< float, int > image_t;
dim3 blck_dim_radon = dim3( 32, 20 );
dim3 grid_dim_radon = dim3( 19, 19 );
dim3 blck_dim_trans = dim3( 32, 20 );
dim3 grid_dim_trans = dim3( 19, 19 );

extern "C" {
   void radon_ga( image_t image, image_t sino );
   void radon_transpose_ga( image_t sino, image_t image );
} //extern "C"

void radon_ga( image_t image, image_t sino )
{
   // Initialize memory on device:
   image.device_init();
   sino.device_init();

   // Copy image data to device:
   image.host2device();

   // Zero out sinogram on device:
   cuda::set<<< 1024, 1024 >>>( sino, 0.0f );
   SYNC_AND_CHECK

   // Initialize golden angles and send to device:
   typedef typename image_t::value_type value_type;
   std::vector< value_type > angles_vector = golden_angles< value_type >( sino.n );
   image_t angles;
   angles.host_data_pointer = &( angles_vector[ 0 ] );
   angles.m = 1;
   angles.n = sino.n;
   angles.device_init();
   angles.host2device();

   // Compute Radon transform on device:
   cuda::radon_general< 110 ><<< grid_dim_radon, blck_dim_radon >>>( image, sino, angles );
   SYNC_AND_CHECK

   // Copy result to host:
   sino.device2host();

   // Free device memory:
   angles.device_destroy();
   sino.device_destroy();
   image.device_destroy();
}

void radon_transpose_ga( image_t sino, image_t image )
{
   // Initialize memory on device:
   image.device_init();
   sino.device_init();

   // Copy sinogram data to device:
   sino.host2device();

   // Zero out image on device:
   cuda::set<<< 1024, 1024 >>>( image, 0.0f );
   SYNC_AND_CHECK

   // Initialize golden angles and send to device:
   typedef typename image_t::value_type value_type;
   std::vector< value_type > angles_vector = golden_angles< value_type >( sino.n );
   image_t angles;
   angles.host_data_pointer = &( angles_vector[ 0 ] );
   angles.m = 1;
   angles.n = sino.n;
   angles.device_init();
   angles.host2device();

   // Compute transpose Radon transform on device:
   cuda::radon_transpose_general< 110 ><<< grid_dim_trans, blck_dim_trans >>>( sino, image, angles );
   SYNC_AND_CHECK

   // Copy result to host:
   image.device2host();

   // Free device memory:
   angles.device_destroy();
   sino.device_destroy();
   image.device_destroy();
}
