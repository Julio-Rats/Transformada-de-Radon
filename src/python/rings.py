import numpy

REGULARIZATION  = 1.5
CGM_NITERATIONS = 1000000
CGM_ERROR       = 0.0000001

##########################################################

def kernel(m, n):
    v = [
        [numpy.array([1, -1]), numpy.array([-3.0 / 2.0, 2, -1 / 2]), numpy.array([-11.0 / 6.0, 3, -3.0 / 2.0, 1.0 / 3.0])],
        [numpy.array([-1, 2, -1]), numpy.array([2, -5, 4, -1])],
        [numpy.array([-1, 3, -3, 1])]
    ]
   
    return v[m - 1][n - 1]  

##########################################################

def ringMatXvec(h, x):

    s = numpy.convolve(x, numpy.flipud(h))
    u = s[numpy.size(h) - 1:numpy.size(x)]
    y = numpy.convolve(u, h)

    return y

##########################################################

def ringCGM(h, alpha, f):

    x0 = numpy.zeros(numpy.size(f))
    r = f - (ringMatXvec(h, x0) + alpha * x0)
    w = -r
    z = ringMatXvec(h, w) + alpha * w
    a = numpy.dot(r, w) / numpy.dot(w, z) 
    x = x0 + numpy.dot(a, w)
    B = 0
	 
    for i in range(CGM_NITERATIONS):
        r = r - numpy.dot(a, z)
        if(numpy.linalg.norm(r) < CGM_ERROR):
	    break
        B = numpy.dot(r, z) / numpy.dot(w, z) 
	w = -r + numpy.dot (B, w) 
	z = ringMatXvec(h, w) + alpha * w
	a = numpy.dot(r, w) / numpy.dot(w, z)
	x = x + numpy.dot(a, w)
    
    return x

##########################################################

def ring_nb(data, m, n): 

    mydata = data

    R = numpy.size(mydata, 0)
    N = numpy.size(mydata, 1)
    	
    #### Removing NaN !!!!!!!!   :-D

    pos = numpy.where(numpy.isnan(mydata) == True)
    mydata[pos] = 0
        
    #### Parameter
        
    alpha = 1 / (2*(mydata.sum(0).max() - mydata.sum(0).min()))

    #### mathematical correction

    pp = mydata.mean(1)

    h = kernel(m, n)             

    #########
    f = -ringMatXvec(h, pp)

    q = ringCGM(h, alpha, f)

    # ## update sinogram

    q.shape = (R, 1)
    K = numpy.kron(q, numpy.ones((1, N)))
    new = numpy.add(mydata, K)
	
    newsino = new.astype(numpy.float32)

    return newsino
	

##########################################################

def ring_b(data, m, n, step): 

    mydata = data

    R = numpy.size(mydata, 0)
    N = numpy.size(mydata, 1)
    	
    #### Removing NaN !!!!!!!!  :-D

    pos = numpy.where(numpy.isnan(mydata) == True)
	
    mydata[pos] = 0

    # ## Kernel & regularization parameter

    h = kernel(m, n)

    # alpha = 1 / (2*(mydata.sum(0).max() - mydata.sum(0).min()))

    #### mathematical correction by blocks

    nblocks = int(N / step)

    new = numpy.ones((R, N))

    for k in range (0, nblocks):

	sino_block = mydata[:, k * step:(k + 1) * step]
	alpha = 1 / (2 * (sino_block.sum(0).max() - sino_block.sum(0).min()))
	pp = sino_block.mean(1)
		
	# #

	f = -ringMatXvec(h, pp)
	q = ringCGM(h, alpha, f)

	# ## update sinogram
        
	q.shape = (R, 1)
	K = numpy.kron(q, numpy.ones((1, step)))
	new[:, k * step:(k + 1) * step] = numpy.add(sino_block, K)

		
    newsino = new.astype(numpy.float32)
    
    return newsino
 
#############################################################

''' Ring Filters without blocks'''

def filter11(SINO):
    return ring_nb(SINO, 1, 1)

def filter12(SINO):
    return ring_nb(SINO, 1, 2)

def filter13(SINO):
    return ring_nb(SINO, 1, 3)

def filter21(SINO):
    return ring_nb(SINO, 2, 1)

def filter22(SINO):
    return ring_nb(SINO, 2, 2)

def filter31(SINO):
    return ring_nb(SINO, 3, 1)


def filter(SINO):

    d1 = ring_nb(SINO, 1, 1)
    d2 = ring_nb(SINO, 2, 1)
    p = d1 * d2
    alpha = REGULARIZATION
    d = numpy.sqrt(p + alpha * numpy.abs(p.min()))
    return d


##########################################################

''' Ring Filters with blocks'''

def filter_block(*args):

    SINO = args[0]
    
    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    d1 = ring_b(SINO, 1, 1, size)
    d2 = ring_b(SINO, 2, 1, size)
    p = d1 * d2
    alpha = REGULARIZATION
	
    d = numpy.sqrt(p + alpha * numpy.fabs(p.min()))
    
    return d


def filter11_block(*args):

    SINO = args[0]
    
    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    return ring_b(SINO, 1, 1, size)


def filter12_block(*args):

    SINO = args[0]

    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    return ring_b(SINO, 1, 2, size)

def filter13_block(*args):

    SINO = args[0]
    
    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    return ring_b(SINO, 1, 3, size)

def filter21_block(*args):

    SINO = args[0]
    
    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    return ring_b(SINO, 2, 1, size)

def filter22_block(*args):

    SINO = args[0]
    
    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    return ring_b(SINO, 2, 2, size)

def filter31_block(*args):

    SINO = args[0]
    
    if len(args) == 1:
	print ("No blocks were given, using 2")
	nblocks = 2
    else:
	nblocks = args[1]

    size = int(SINO.shape[0] / nblocks)
	
    return ring_b(SINO, 3, 1, size)
    
##########################################################

