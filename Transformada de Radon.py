import tmp.ray_tracing_cuda_float as rt
import numpy as np
import matplotlib.pyplot as pp
from PIL import Image

img = np.array(Image.open('phantom_sheep_logan.png').resize((512, 512)).convert('L'))
# img = np.array(Image.open('two_squares.png').resize((512, 512)).rotate(-45).convert('L'))
#img = np.array(Image.open('space.jpg').resize((512, 256)).convert('L'))

img = (img - img.min())/(img.max()-img.min())

# pp.figure( figsize = ( 16, 8 ) )
# pp.imshow( img )
# pp.show()

( radon, radon_t ) = rt.make_radon_transp( ( 1024, 512 ), ( 0.0, 1.0 ), ( np.pi, -1.0 ), img.shape )

sino = radon( img )

pp.figure( figsize = ( 16, 8 ) )
pp.imshow( sino, extent = ( 0.0, np.pi, -1.0, 1.0 ) )
pp.show()

def EMM(sino, radon, radon_t, img_shape, sino_shape=( 2048, 1024 ), eps=1e-3, maxIter=1e3):
    new_image = np.ones(img_shape)
    matrix_p  = radon_t(np.ones(sino_shape))
    sino_new  = radon(new_image)
    erro      = float(np.sum(np.abs(sino-sino_new)))
    itera     = 0
    converg   = [erro]
    minErro   = eps*sino_shape[0]*sino_shape[1]

    if np.count_nonzero(matrix_p) != np.ma.count(matrix_p):
        matrix_p = matrix_p + 1e-9

    while (erro > minErro) and (itera < maxIter):
        itera = itera + 1
        print(f"Iteração {itera} - MaxInt {maxIter} | Global Erro = {format(erro, '.5f')} | Mínimo Estimado {format(minErro,'.5f')}", end="\r")
        if np.count_nonzero(sino_new) != np.ma.count(sino_new):
            sino_new = sino_new + 1e-12

        matrix_n  = radon_t(sino/sino_new)
        new_image = new_image*(matrix_n/matrix_p)

        sino_new     = radon(new_image)
        erro         = float(np.sum(np.abs(sino-sino_new)))
        converg.append(erro)

    print(f"Iteração {itera} - MaxInt {maxIter} | Global Erro = {format(erro, '.5f')}", end="\n")
    return (new_image, converg)


(img_emm, converg) = EMM(sino, radon, radon_t, img.shape, sino.shape, eps=1e-4, maxIter=300)

pp.figure( figsize = ( 16, 8 ) )
pp.imshow( img_emm )
pp.show()
