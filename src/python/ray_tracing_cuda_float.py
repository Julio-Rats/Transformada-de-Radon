# I use the one encoding: utf8
import ctypes
import ctypes.util
import numpy
import math
import tmp.pyraft_noc as pr

# Load required libraies:
libcudart = ctypes.CDLL( ctypes.util.find_library( "cudart" ), mode=ctypes.RTLD_GLOBAL )
libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
libtomo   = ctypes.CDLL( "./tmp/cuda_radon.so" )

# "float *" type:
_c_float_p = ctypes.POINTER( ctypes.c_float )

class RAFT_MATRIX( ctypes.Structure ):
   """A raft_matrix from raft:"""
   _fields_ = [ ( "p_data_device", _c_float_p ),
                ( "p_data_host", _c_float_p ),
                ( "lines", ctypes.c_int ),
                ( "columns", ctypes.c_int )
             ]

class RAFT_IMAGE( ctypes.Structure ):
   """A raft_image from raft:"""
   _fields_ = [ ( "data", RAFT_MATRIX ),
                ( "tl_x", ctypes.c_float ),
                ( "tl_y", ctypes.c_float ),
                ( "br_x", ctypes.c_float ),
                ( "br_y", ctypes.c_float )
              ]

def make_RAFT_MATRIX( array ):
   """Mak a raft_matrix from a numpy.ndarray"""
   return RAFT_MATRIX( ctypes.cast( 0, _c_float_p ),
                       ctypes.cast( array.ctypes.data, _c_float_p ),
                       ctypes.c_int( array.shape[ 0 ] ),
                       ctypes.c_int( array.shape[ 1 ] )
                      )

def make_RAFT_IMAGE( array, top_left = ( -1.0, 1.0 ), bottom_right = ( 1.0, -1.0 ) ):
   """Make a raft_matrix from a numpy.ndarray from a pyraft.image or from a pyraft.RAFT_MATRIX"""
   if isinstance( array, numpy.ndarray ):
      return RAFT_IMAGE( make_RAFT_MATRIX( array ),
                         ctypes.c_float( top_left[ 0 ] ),
                         ctypes.c_float( top_left[ 1 ] ),
                         ctypes.c_float( bottom_right[ 0 ] ),
                         ctypes.c_float( bottom_right[ 1 ] )
                       )
   elif isinstance( array, RAFT_MATRIX ):
      return RAFT_IMAGE( array,
                         ctypes.c_float( top_left[ 0 ] ),
                         ctypes.c_float( top_left[ 1 ] ),
                         ctypes.c_float( bottom_right[ 0 ] ),
                         ctypes.c_float( bottom_right[ 1 ] )
                       )
   elif isinstance( array, image ):
      return RAFT_IMAGE( array,
                         ctypes.c_float( array.top_left[ 0 ] ),
                         ctypes.c_float( array.top_left[ 1 ] ),
                         ctypes.c_float( array.bottom_right[ 0 ] ),
                         ctypes.c_float( array.bottom_right[ 1 ] )
                       )

# Function prototypes:
libtomo.radon.argtypes = [ RAFT_IMAGE, RAFT_IMAGE ]
libtomo.radon_transpose.argtypes = [ RAFT_IMAGE, RAFT_IMAGE ]

def make_radon_transp(
                       sino_shape, sino_top_left = None, sino_bottom_right = None,
                       img_shape = None, img_top_left = None, img_bottom_right = None
                     ):

   # Sinogram geometry:
   # If sinogram given, extract what we need:
   if isinstance( sino_shape, pr.image ):
      # Avoid redundancies
      if ( not ( sino_top_left is None ) ) or ( not ( sino_bottom_right is None ) ):
         raise ValueError( 'Do not provide redundant sinogram extents!' )
      # Extract geometry:
      sino_top_left = sino_shape.top_left
      sino_bottom_right = sino_shape.bottom_right
      sino_shape = sino_shape.shape
   else:
      # If sinogram not given, use provided geometry or defaults:
      if sino_top_left is None:
         sino_top_left = ( 0.0, 1.0 - 1.0 / float( sino_shape[ 0 ] ) )
      if sino_bottom_right is None:
         sino_bottom_right = (
                               #math.pi * ( 1.0 - 1.0 / float( sino_shape[ 1 ] ) ),
                               math.pi,
                               -1.0 + 1.0 / float( sino_shape[ 0 ] )
                             )

   # Image geometry:
   if img_shape is None:
      img_shape = ( sino_shape[ 0 ], sino_shape[ 0 ] )
   if img_top_left is None:
      img_top_left = ( -1.0, 1.0 )
   if img_bottom_right is None:
      img_bottom_right = ( 1.0, -1.0 )

   def radon( x ):
      """
         Compute projection through ray-tracing techniques
      """
      sino_data = numpy.zeros( sino_shape, dtype = 'float32', order = 'F' )
      SINO = make_RAFT_IMAGE( sino_data, sino_top_left, sino_bottom_right )

      x = x.astype( 'float32', order = 'F' )
      IMAGE = make_RAFT_IMAGE( x, img_top_left, img_bottom_right )

      libtomo.radon( IMAGE, SINO )

      return pr.image( sino_data.astype( 'float64', order = 'C' ), sino_top_left, sino_bottom_right )

   def radon_transpose( y ):
      """
         Compute backprojection through ray-tracing techniques
      """
      image_data = numpy.zeros( img_shape, dtype = 'float32', order = 'F' )
      IMAGE = make_RAFT_IMAGE( image_data, img_top_left, img_bottom_right )

      y = y.astype( 'float32', order = 'F' )
      SINO = make_RAFT_IMAGE( y, sino_top_left, sino_bottom_right )

      libtomo.radon_transpose( SINO, IMAGE )

      return pr.image( image_data.astype( 'float64', order = 'C' ), img_top_left, img_bottom_right )

   return radon, radon_transpose
