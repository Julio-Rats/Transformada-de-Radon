#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

namespace cuda {

   /// Type-safe Archimedes' constant.
   /**
    * Simple class providing a type-safe estimate for pi.
    */
   template < class value_t >
   struct PI {
      /// Value of pi.
      /**
       * \return Approximate value of pi in type value_t.
       */
      inline __device__ __host__
      static value_t value ( void )
      {
         return( static_cast< value_t >( 3.14159265358979 ) );
      }
   };

   /// Atomic addition for double precision.
   /**
    * Follow CUDA C Programming Guide in order to implement atomic
    * addition of double precision numbers.
    * 
    * \return Value previously contained in memory.
    * 
    * \param address Memory address where to add value;
    * \param val Value to be added.
    */
   inline __device__
   double atomicAdd( double * address, double val )
   {
      //Code adapted from NVIDIA CUDA C Programming guide.

      unsigned long long *address_as_ull = reinterpret_cast< unsigned long long * >( address );
      unsigned long long new_value, old_value = *address_as_ull;

      do {
         new_value = old_value;
         old_value = atomicCAS(
            address_as_ull,
            new_value,
            __double_as_longlong(
               val + __longlong_as_double( new_value )
            )
         );
         // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
      } while ( new_value != old_value );

      return( __longlong_as_double( old_value ) );
   }

   /// Atomic addition for single precision.
   /**
    * Use CUDA C atomic addition intrinsics.
    * 
    * \return Value previously contained in memory.
    * 
    * \param address Memory address where to add value;
    * \param val Value to be added.
    */
   inline __device__
   float atomicAdd( float * address, float val )
   {
      return ::atomicAdd( address, val );
   }
} // namespace cuda

#endif // #ifndef CUDA_UTILS_H
