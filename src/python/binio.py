import numpy
import matplotlib.pyplot as pp

def load_bin( filename, dtype = 'float64' ):

   with open( filename, 'rb' ) as f:
      shape = numpy.fromstring( f.read( 8 ), dtype = 'int32' )
      array = numpy.fromfile( f, dtype = dtype ).reshape( shape, order = 'C' )

   return array

def save_bin( filename, array, dtype = None ):

   if dtype is None:
      dtype = array.dtype

   with open( filename, 'wb' ) as f:
      f.write( numpy.array( array.shape, dtype = 'int32' ).tostring() )
      f.write( array.astype( dtype ).tostring() )
