#We use the one encoding: utf8
import math
import numpy

class image( numpy.ndarray ):
   """This class represents an image. This may not be the ideal foundation,
   because there are already some options for image classes. Further study is necessary."""

   def __new__(
                subtype,
                shape,
                top_left = None,
                bottom_right = None,
                extent = None,
                x_extent = None,
                y_extent = None,
                **kwargs
              ):
      """Creates and returns a new object of the correct subtype"""

      # Which field-of-view arguments where given?
      extent_given = extent or x_extent or y_extent
      corner_given = top_left or bottom_right

      # Are they acceptable?
      if extent_given and corner_given:
         raise TypeError( 'Mutually exclusive arguments given.' )

      # Extent given, adjust corners:
      if extent_given:
         # Extent given by parts:
         if not extent:
            if not x_extent:
               x_extent = ( -1.0, 1.0 )
            if not y_extent:
               y_extent = ( -1.0, 1.0 )
         # Extent given fully:
         else:
            x_extent = ( extent[ 0 ], extent[ 1 ] )
            y_extent = ( extent[ 2 ], extent[ 3 ] )
         # Finally, we can set up corners:
         top_left     = ( x_extent[ 0 ], y_extent[ 1 ] )
         bottom_right = ( x_extent[ 1 ], y_extent[ 0 ] )

      # pyraft.image given as argument
      if isinstance( shape, image ):

         # Check for given corners:
         if not extent_given:
            if not top_left:
               top_left = shape.top_left
            if not bottom_right:
               bottom_right = shape.bottom_right

         # No arguments other than corners can be taken:
         if kwargs:
            raise TypeError( 'Unhandled arguments!' )

         ## In here, shape is actually a pyraft.image:
         #obj = numpy.asarray( shape ).view( subtype )
         # TODO: No view, make a copy! But there must be a neater way...
         obj = numpy.ndarray.__new__( subtype, shape.shape, **kwargs )
         obj[ ... ] = shape[ ... ]

      else:

         # Check for given corners:
         if not extent_given:
            if not top_left:
               top_left = ( -1.0, 1.0 )
            if not bottom_right:
               bottom_right = ( 1.0, -1.0 )

         # numpy.ndarray given as argument:
         if isinstance( shape, numpy.ndarray ):

            if kwargs:
            # No arguments other than corners can be taken:
               raise TypeError( 'Unhandled arguments!' )

            # In here, shape is actually a numpy.ndarray:
            #obj = numpy.asarray( shape ).view( subtype )
            # TODO: No view, make a copy! But there must be a neater way...
            obj = numpy.ndarray.__new__( subtype, shape.shape, **kwargs )
            obj[ ... ] = shape[ ... ]

         # We must create a zero array:
         else:

            # Default data type is double:
            if not ( 'dtype' in kwargs ):
               kwargs[ 'dtype' ] = numpy.float64
            obj = numpy.ndarray.__new__( subtype, shape, **kwargs )
            obj[ : ] = 0.0

      # All relevant dimensions must match:
      if ( len( obj.shape ) != len( top_left ) ) or ( len( top_left ) != len( bottom_right ) ):
         raise TypeError( 'Dimensions must match!' )

      # Set new attributes:
      obj.top_left = top_left
      obj.bottom_right = bottom_right
      try:
         obj.sampling_distances = ( ( bottom_right[ 0 ] - top_left[ 0 ] ) / ( obj.shape[ 1 ] - 1.0 ),
                                    ( bottom_right[ 1 ] - top_left[ 1 ] ) / ( obj.shape[ 0 ] - 1.0 )
                                  )
      except ZeroDivisionError:
         obj.sampling_distances = ( 0.0, 0.0 )
      return obj

   def __array_finalize__( self, obj ):
      """Set self attributes"""
      if obj is None: return # When ran from __new__

      # Else do the job:
      self.top_left = getattr( obj, 'top_left', None )
      self.bottom_right = getattr( obj, 'bottom_right', None )
      self.sampling_distances = getattr( obj, 'sampling_distances', None )

   def __reduce__( self ):

      # Initial state is only ndarray state:
      full_state = list( numpy.ndarray.__reduce__( self ) )

      #Further attributes:
      image_state = ( self.top_left, self.bottom_right, self.sampling_distances )

      # Add image attributes:
      full_state[ 2 ] = ( full_state[ 2 ], image_state )

      return tuple( full_state )

   def __setstate__( self, state ):

      # Call superclass' __setstate__:
      numpy.ndarray.__setstate__( self, state[ 0 ] )

      # Set our own state:
      self.top_left = state[ 1 ][ 0 ]
      self.bottom_right = state[ 1 ][ 1 ]
      self.sampling_distances = state[ 1 ][ 2 ]

   def sample_coordinates( self, idx ):
      """ Returns coordinates of sample """
      return ( self.top_left[ 0 ] + idx[ 1 ] * self.sampling_distances[ 0 ], self.top_left[ 1 ] + idx[ 0 ] * self.sampling_distances[ 1 ] )

   def get_y_coordinate( self, idx ):
      """ Returns y-coordinate of row """
      return self.top_left[ 1 ] + idx * self.sampling_distances[ 1 ]

   def get_x_coordinate( self, idx ):
      """ Returns x-coordinate of column """
      return self.top_left[ 0 ] + idx * self.sampling_distances[ 0 ]

   # Extent:
   def extent( self ):
      return ( self.top_left[ 0 ], self.bottom_right[ 0 ], self.bottom_right[ 1 ], self.top_left[ 1 ] )
