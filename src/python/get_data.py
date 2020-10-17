import lnls_data as ld
import binio
import sys
import rings

path = sys.argv[ 1 ]
slice_number = int( sys.argv[ 2 ] )
if len( sys.argv ) > 4:
   rotation = int( sys.argv[ 4 ] )
else:
   rotation = None

( count, dark, flat, sino ) = ld.slice_extract( path, slice_number, rotation = rotation )

binio.save_bin( 'count.data', count )
binio.save_bin( 'dark.data', dark )
binio.save_bin( 'flat.data', flat )
if ( len( sys.argv ) > 3 ) and ( sys.argv[ 3 ] in [ 'True', 'true' ] ):
   sino  = rings.filter( sino ).astype( 'float64' )
binio.save_bin( 'sino.data', sino )
