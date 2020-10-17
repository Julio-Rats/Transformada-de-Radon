#include "cuda_radon.h"
#include "cuda_image2.h"

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
   void radon( image_t image, image_t sino );
   void radon_transpose( image_t sino, image_t image );
} //extern "C"

void radon( image_t image, image_t sino )
{
   // Initialize memory on device:
   image.device_init();
   sino.device_init();

   // Copy image data to device:
   image.host2device();

   // Zero out sinogram on device:
   cuda::set<<< 1024, 1024 >>>( sino, 0.0f );
   SYNC_AND_CHECK

   // Compute Radon transform on device:
   cuda::radon< 110 ><<< grid_dim_radon, blck_dim_radon >>>( image, sino );
   SYNC_AND_CHECK

   // Copy result to host:
   sino.device2host();


   // Free device memory:
   sino.device_destroy();
   image.device_destroy();
}

void radon_transpose( image_t sino, image_t image )
{
   // Initialize memory on device:
   image.device_init();
   sino.device_init();

   // Copy sinogram data to device:
   sino.host2device();

   // Zero out image on device:
   cuda::set<<< 1024, 1024 >>>( image, 0.0f );
   SYNC_AND_CHECK

   // Compute transpose Radon transform on device:
   cuda::radon_transpose< 110 ><<< grid_dim_trans, blck_dim_trans >>>( sino, image );
   SYNC_AND_CHECK

   // Copy result to host:
   image.device2host();

   // Free device memory:
   sino.device_destroy();
   image.device_destroy();
}
