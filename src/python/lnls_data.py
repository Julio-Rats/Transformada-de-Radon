import h5py
import numpy

def slice_extract( path, slice_number, area_mean = None, rotation = None ):

   # Open data file for reading:
   f = h5py.File( path + 'tomo.h5', 'r' )

   # Associate data set:
   data = f[ 'images' ]

   # Get flat and dark data:
   f_dark_b = h5py.File( path + 'tomo_dark_before.h5', 'r' )
   f_flat_b = h5py.File( path + 'tomo_flat_before.h5', 'r' )
   dark_data_b = f_dark_b[ 'darks' ]
   flat_data_b = f_flat_b[ 'flats' ]
   dark_b = numpy.reshape( numpy.array( dark_data_b[ 0, slice_number, : ] ).astype( 'double' ), ( 2048, ) )
   flat_b = numpy.reshape( numpy.array( flat_data_b[ 0, slice_number, : ] ).astype( 'double' ), ( 2048, ) )

   f_flat_a = h5py.File( path + 'tomo_flat_after.h5', 'r' )
   flat_data_a = f_flat_a[ 'flats' ]
   flat_a = numpy.reshape( numpy.array( flat_data_a[ 0, slice_number, : ] ).astype( 'double' ), ( 2048, ) )
   if not ( area_mean is None ):
      mean_count = []
      for i in range( data.shape[ 0 ] ):
         mean_count.append( numpy.mean( data[ i, area_mean[ 0 ] : area_mean[ 1 ], area_mean[ 2 ] : area_mean[ 3 ] ] ) )

   # Open image data:
   im = numpy.transpose( numpy.array( data[ :, slice_number, : ] ).astype( 'double' ) )
   flat = numpy.zeros( im.shape )
   dark = numpy.zeros( im.shape )

   for j in range( data.shape[ 0 ] ):
      alpha = float( j ) / float( data.shape[ 0 ] )
      if area_mean is None:
         curr_flat = alpha * flat_a + ( 1 - alpha ) * flat_b
      else:
         curr_flat = flat_b * mean_count[ j ] / mean_count[ 0 ]
      flat[ :, j ] = curr_flat
      dark[ :, j ] = dark_b

   sino = numpy.log( ( flat - dark ) / ( im - dark ) )

   mn = float( 'inf' )
   if rotation is None:
      rotation = 0
      for rot in range( -100, 101 ):
         diff = numpy.sum( numpy.abs( numpy.roll( sino[ :, 0 ], rot ) - numpy.flip( numpy.roll( sino[ :, -1 ], rot ), 0 ) ) )
         if diff < mn:
            mn = diff
            rotation = rot

   im = numpy.roll( im, rotation, axis = 0 )
   dark = numpy.roll( dark, rotation, axis = 0 )
   flat = numpy.roll( flat, rotation, axis = 0 )
   sino = numpy.roll( sino, rotation, axis = 0 )

   return ( im, dark, flat, sino )
