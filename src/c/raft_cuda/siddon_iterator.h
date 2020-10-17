#ifndef SIDDON_ITERATOR
#define SIDDON_ITERATOR

#ifndef __CUDA_ARCH__
#define __DEVICE__
#include <cmath>
#include <algorithm>
#include <limits>
#else
#define __DEVICE__ __device__
#endif // #ifndef __CUDA_ARCH__

/**
 * \~english   \file siddon_iterator.h Defines a template class for a fast-Siddon ray tracer.
 * \~brazilian \file siddon_iterator.h Define um modelo de classe para um traçador de Siddon rápido.
 */

namespace rtt { // rtt = ray-tracing for tomography

#ifndef __CUDA_ARCH__
   template < class T >
   class limits {

      public:

         static inline __DEVICE__ T max( void )
         {
            return( std::numeric_limits< T >::max() );
         }
   };
#else
   template < class T >
   class limits {
   };
   class limits < double > {

      public:

         static inline __DEVICE__ double max( void )
         {
            return( 1.7976931348623158e+308 );
         }
   };
   class limits < float > {

      public:

         static inline __DEVICE__ float max( void )
         {
            return( 3.402823466e+38f );
         }
   };
#endif

   /// \~english Siddon Iterator. \~brazilian Iterador de Siddon
   /**
    * \~english   This class performs steps of improved Siddon's algorithm at each application of raft::siddon_iterator.operator++(). You should not have to initialize it by hand, use raft::image.begin() to do the job instead.
    * \~brazilian Esta classe, se corretamente inicializada, executa um passo do algoritmo de Siddon a cada aplicação de raft::siddon_iterator.operator++(). Você não deveria ter que inicializar um objeto manualmente, utilize raft::image.begin() para fazer o serviço.
    *
    * \~english   \tparam value_t Floating point type;
    * \~brazilian \tparam value_t Tipo de ponto flutuante;
    *
    * \~english   \tparam index_t Index type (integer);
    * \~brazilian \tparam index_t Tipo dos índices (inteiro);
    *
    * \~english   \tparam diff_t Difference type (signed integer).
    * \~brazilian \tparam diff_t Tipo de diferença (inteiro com sinal).
    */
   template< class value_t = double, class index_t = int, class diff_t = int >
   struct siddon_iterator {

      value_t eps_; ///< \~english Current intersection length. \~brazilian Comprimento da intersecção atual.

      index_t j_; ///< \~english Current column index. \~brazilian Índice da coluna atual.
      index_t i_; ///< \~english Current line index. \~brazilian Índice da linha atual.

      index_t n_; ///< \~english Number of columns. \~brazilian Número de colunas.
      index_t m_; ///< \~english Number of linhas. \~brazilian Número de linhas.

      value_t tl_x_; ///< \~english Top-left x-coordinate of the image. \~brazilian Coordenada x do canto superior esquerdo da imagem.
      value_t tl_y_; ///< \~english Top-left y-coordinate of the image. \~brazilian Coordenada y do canto superior esquerdo da imagem.

      value_t br_x_; ///< \~english Bottom-right x-coordinate of the image. \~brazilian Coordenada x do canto inferior direito da imagem.
      value_t br_y_; ///< \~english Bottom-right y-coordinate of the image. \~brazilian Coordenada y do canto inferior direito da imagem.

      value_t ell_x_; ///< \~english Pixel width. \~brazilian Largura do pixel.
      value_t ell_y_; ///< \~english Pixel heigh. \~brazilian Altura do pixel.

      diff_t nabla_j_; ///< \~english Column directions. \~brazilian Direções das colunas.
      diff_t nabla_i_; ///< \~english Line directions. \~brazilian Direções das linhas.

      value_t cos_theta_; ///< \~english Current angle's cosine. \~brazilian Cosseno do ângulo atual.
      value_t sin_theta_; ///< \~english Current angle's sine. \~brazilian Seno do ângulo atual.

      value_t lambda_; ///< \~english Current position in the line segment. \~brazilian Posição atual no segmento de linha.

      value_t lambda_j_; ///< \~english Position of the intersection of the line segment with vertical pixels border. \~brazilian Posição da intersecção do segmento com a fronteira vertical do pixel.
      value_t lambda_i_; ///< \~english Position of the intersection of the line segment with horizontal pixels border. \~brazilian Posição da intersecção do segmento com a fronteira horizontal do pixel.

      value_t Delta_j_; ///< \~english Distance between vertical parallel borders (as measured by running the segment). \~brazilian Distância entre fronteiras paralelas verticais, conforme medidas através do caminho de integração.
      value_t Delta_i_; ///< \~english Distance between horizontal parallel borders (as measured by running the segment). \~brazilian Distância entre fronteiras paralelas horizontais, conforme medidas através do caminho de integração.

      index_t xi_; ///< \~english Current axis intersection. \~brazilian Eixo de intersecção atual.
      index_t not_xi_;

      value_t delta_; ///< \~english Current intersection length. \~brazilian Comprimento da intersecção atual.

      template< class T >
      inline __DEVICE__ diff_t sign_( T x )
      {
         return( ( x >= T( 0 ) ) - ( x <= T( 0 ) ) );
      }

      inline __DEVICE__ void update_params_();

#ifdef __CUDA_ARCH__
      inline __DEVICE__ value_t min_( value_t x, value_t y )
      {
         return( fmin( x, y ) );
      }
      inline __DEVICE__ value_t max_( value_t x, value_t y )
      {
         return( fmax( x, y ) );
      }
      inline __DEVICE__ value_t abs_( value_t x )
      {
         return( abs( x ) );
      }
      inline __DEVICE__ value_t floor_( value_t x )
      {
         return( floor( x ) );
      }
      inline __DEVICE__ value_t cos_( value_t x )
      {
         return( cos( x ) );
      }
      inline __DEVICE__ value_t sin_( value_t x )
      {
         return( sin( x ) );
      }
#else
      inline __DEVICE__ value_t min_( value_t x, value_t y )
      {
         return( std::min( x, y ) );
      }
      inline __DEVICE__ value_t max_( value_t x, value_t y )
      {
         return( std::max( x, y ) );
      }
      inline __DEVICE__ value_t abs_( value_t x )
      {
         return( std::abs( x ) );
      }
      inline __DEVICE__ value_t floor_( value_t x )
      {
         return( std::floor( x ) );
      }
      inline __DEVICE__ value_t cos_( value_t x )
      {
         return( std::cos( x ) );
      }
      inline __DEVICE__ value_t sin_( value_t x )
      {
         return( std::sin( x ) );
      }
#endif // #ifdef __CUDA_ARCH__

   public:

      typedef value_t value_type; ///< \~english Type for floating point values.\~brazilian Tipo dos valores de ponto flutuante.
      typedef index_t index_type; ///< \~english Type for indices.\~brazilian Tipo dos índices.
      typedef diff_t difference_type; ///< \~english Type for differences between indices.\~brazilian Tipo das diferenças entre índices.

      inline __DEVICE__
      void set_image(
         index_t m, index_t n,
         value_t tl_x = value_t( -1 ),
         value_t tl_y = value_t(  1 ),
         value_t br_x = value_t(  1 ),
         value_t br_y = value_t( -1 )
      );
      inline __DEVICE__
      void set_theta( value_t theta );
      inline __DEVICE__
      void set_t( value_t t );

      inline __DEVICE__
      siddon_iterator< value_t, index_t, diff_t >& operator++( void );

      inline __DEVICE__
      bool valid( void ) const
      {
         return(
            ( i_ >= index_t( 0 ) ) && ( i_ < m_ ) &&
            ( j_ >= index_t( 0 ) ) && ( j_ < n_ )
         );
      }

      inline __DEVICE__
      index_t i( void ) const
      {
         return( i_ );
      }
      inline __DEVICE__
      index_t j( void ) const
      {
         return( j_ );
      }

      inline __DEVICE__
      value_t delta( void ) const
      {
         return( delta_ );
      }
   };

   template< class value_t, class index_t, class diff_t >
   inline __DEVICE__
   void siddon_iterator< value_t, index_t, diff_t >::set_image(
         index_t m,
         index_t n,
         value_t tl_x,
         value_t tl_y,
         value_t br_x,
         value_t br_y
      )
   {
      m_ = m;
      n_ = n;
      tl_x_ = tl_x;
      tl_y_ = tl_y;
      br_x_ = br_x;
      br_y_ = br_y;
      ell_x_ = ( br_x - tl_x ) / n;
      ell_y_ = ( br_y - tl_y ) / m;
   }

   template< class value_t, class index_t, class diff_t >
   inline __DEVICE__
   void siddon_iterator< value_t, index_t, diff_t >::set_theta( value_t theta )
   {
      cos_theta_= cos_( theta );
      sin_theta_= sin_( theta );

      Delta_i_ = cos_theta_ ?
         abs_( ell_y_ / cos_theta_ ) :
         limits< value_t >::max();

      Delta_j_ = sin_theta_ ?
         abs_( ell_x_ / sin_theta_ ) :
         limits< value_t >::max();

      nabla_i_ =  sign_( ell_y_ ) * sign_( cos_theta_ );
      nabla_j_ = -sign_( ell_x_ ) * sign_( sin_theta_ );
   }

#define SET_IJ( \
   index_a, /* j_ / i_ */ \
   index_b, /* i_ / j_ */ \
   alpha_a, /* cos_theta_ / sin_theta_ */ \
   beta_a, /* (-sin_theta_) / cos_theta_ */ \
   alpha_b, /* sin_theta_ / cos_theta_ */ \
   beta_b, /* cos_theta_ / (-sin_theta_) */ \
   tl_a, /* tl_x_ / tl_y_ */ \
   tl_b, /* tl_y_ / tl_x_ */ \
   br_a, /* br_x_ / br_y_ */ \
   br_b, /* br_y_ / br_x_ */ \
   ell_a, /* ell_x_ / ell_y_ */ \
   ell_b, /* ell_y_ / ell_x_ */ \
   m_a, /* n_ / m_ */ \
   m_b, /* m_ / n_ */ \
   nabla_a, /* nabla_j_ / nabla_i_ */ \
   nabla_b /* nabla_i_ / nabla_j_ */ \
) \
/* Compute parameter value of first intersection with \
   horizontal image boundary: */ \
temp_1 = ( tl_b - ( t * alpha_b ) ) / beta_b; \
temp_2 = ( br_b - ( t * alpha_b ) ) / beta_b; \
lambda_ = min_( temp_1, temp_2 ); \
\
/* X-coordinate of intersection with horizontal boundary: */ \
coord = ( t * alpha_a ) + ( lambda_ * beta_a ); \
\
/* Is coordinate inside image or is line straight? */ \
if ( \
   ( ( coord >= min_( tl_a, br_a ) ) && ( coord <= max_( tl_a, br_a ) ) ) || \
   ( beta_a == value_t( 0.0 ) ) \
   ) \
{ \
   /* Index of starting line: */ \
   /* index_b = index_t( 0 ) + ( ( temp_1 > temp_2 ) * ( m_b - index_t( 1 ) ) ); */ \
   index_b = ( temp_1 > temp_2 ) * ( m_b - index_t( 1 ) ); \
\
   /* Column index: */ \
   temp_1 = ( coord - tl_a ) / ell_a; \
   temp_2 = floor( temp_1 ); \
   /* If is integer, need to correct depending on integration direction: */ \
   temp_1 -= ( ( temp_1 == temp_2 ) && ( nabla_a < diff_t( 0 ) ) ); \
   /* Round to integer: */ \
   index_a = temp_2; \
} \
else \
{ \
   /* We need to perform some computations unstably. */ \
\
   /* Compute parameter value of first intersection with \
      vertical image boundary: */ \
   temp_1 = ( tl_a - ( t * alpha_a ) ) / beta_a; \
   temp_2 = ( br_a - ( t * alpha_a ) ) / beta_a; \
   lambda_ = min_( temp_1, temp_2 ); \
\
   /* Y-coordinate of intersection with vertical boundary: */ \
   coord = ( t * alpha_b ) + ( lambda_ * beta_b ); \
\
   /* Is coordinate inside image? */ \
   if ( ( coord >= min_( tl_b, br_b ) ) && ( coord <= max_( tl_b, br_b ) ) ) \
   {\
      /* Index of starting column: */ \
      index_a = index_t( 0 ) + ( ( temp_1 > temp_2 ) * ( m_a - index_t( 1 ) ) ); \
\
      /* Line index: */ \
      temp_1 = ( coord - tl_b ) / ell_b; \
      temp_2 = floor( temp_1 ); \
      /* If is integer, need to correct depending on integration direction: */ \
      temp_1 -= ( ( temp_1 == temp_2 ) && ( nabla_b < diff_t( 0 ) ) ); \
      /* Round to integer: */ \
      index_b = temp_2; \
   } \
   else \
   { \
      /* Select a pixel outside image range and leave: */ \
      index_a = m_a; \
      index_b = m_b; \
      return; \
   } \
}

   template< class value_t, class index_t, class diff_t >
   inline __DEVICE__
   void siddon_iterator< value_t, index_t, diff_t >::update_params_( void )
   {
      // Which will be the intersection parameter?
      xi_     = ( lambda_j_ <= lambda_i_ );
      not_xi_ = 1 - xi_;

      // Intersection length:
      delta_ = -lambda_;
      lambda_ = min_( lambda_i_, lambda_j_ );
      delta_ += lambda_;

      // Update next intersection parameters:
      lambda_i_ += not_xi_ * Delta_i_;
      lambda_j_ += xi_     * Delta_j_;
   }

   template< class value_t, class index_t, class diff_t >
   inline __DEVICE__
   void siddon_iterator< value_t, index_t, diff_t >::set_t( value_t t )
   {
      // Temporaries for computations:
      value_t temp_1, temp_2, coord;

      if ( abs_( cos_theta_ ) > abs_( sin_theta_ ) )
      {
         SET_IJ(
            j_, i_,
            cos_theta_, (-sin_theta_),
            sin_theta_, cos_theta_,
            tl_x_, tl_y_,
            br_x_, br_y_,
            ell_x_, ell_y_,
            n_, m_,
            nabla_j_, nabla_i_
         );
      }
      else
      {
         SET_IJ(
            i_, j_,
            sin_theta_, cos_theta_,
            cos_theta_, (-sin_theta_),
            tl_y_, tl_x_,
            br_y_, br_x_,
            ell_y_, ell_x_,
            m_, n_,
            nabla_i_, nabla_j_
         );
      }

      // Determine next intersection with vertical
      // pixel boundary:
      coord = tl_x_ + ( ell_x_ * ( j_ + ( nabla_j_ > diff_t( 0 ) ) ) );
      lambda_j_ = sin_theta_ ?
         ( coord - ( t * cos_theta_ ) ) / ( -sin_theta_ ) :
         limits< value_t >::max();

      // Determine next intersection with horizontal
      // pixel boundary:
      coord = tl_y_ + ( ell_y_ * ( i_ + ( nabla_i_ > diff_t( 0 ) ) ) );
      lambda_i_ = cos_theta_ ?
         ( coord - ( t * sin_theta_ ) ) / cos_theta_ :
         limits< value_t >::max();

      // Update parameters:
      update_params_();
      delta_ = max_( delta_, static_cast< value_t >( 0.0 ) );
   }

#undef SET_IJ

   template< class value_t, class index_t, class diff_t >
   inline __DEVICE__
   siddon_iterator< value_t, index_t, diff_t >&
   siddon_iterator< value_t, index_t, diff_t >::operator++( void )
   {
      // Update pixel:
      i_ += not_xi_ * nabla_i_;
      j_ += xi_ * nabla_j_;

      // Update parameters:
      update_params_();

      return( *this );
   }

} // namespace rtt
#undef __DEVICE__

#endif // #ifndef SIDDON_ITERATOR
