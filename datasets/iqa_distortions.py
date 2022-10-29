import numpy as np
from PIL import Image, ImageFilter
import skimage.morphology
from scipy import ndimage
import random
from skimage import color,filters,io
from sklearn.preprocessing import normalize
import io
from scipy.interpolate import UnivariateSpline
import PIL
from scipy import interpolate
import skimage

import warnings
warnings.filterwarnings("ignore")

#dist_level = [0,1,2,3,4]


def curvefit (xx, coef):


    x = np.array([0,0.5,1])
    y = np.array([0,coef,1])

    tck = UnivariateSpline(x, y, k=2)
    return np.clip(tck(xx),0,1)


def mapmm(e):

    mina = 0.0
    maxa = 1.0
    minx = np.min(e)
    maxx = np.max(e)
    if minx<maxx : 
        e = (e-minx)/(maxx-minx)*(maxa-mina)+mina
    return e

def imblurgauss(im, level):
    # Takes in PIL Image and returns Gaussian Blurred PIL Image
    levels = [0.1, 0.5, 1, 2, 5]
    sigma = levels[level]

    im_dist = im.filter(ImageFilter.GaussianBlur(radius = sigma))
    return im_dist


def imblurlens(im, level):
    # Takes PIL Image and returns lens blurred image

    # MATLAB version https://github.com/alexandrovteam/IMS_quality/blob/master/codebase/fspecialIM.m
    levels = [1, 2, 4, 6, 8]
    radius = levels[level]

    im = np.array(im)
    crad  = int(np.ceil(radius-0.5))
    [x,y] = np.meshgrid(np.arange(-crad,crad+1,1), np.arange(-crad,crad+1,1), indexing='xy')
    maxxy = np.maximum(abs(x),abs(y))
    minxy = np.minimum(abs(x),abs(y))
    m1 = np.multiply((radius**2 <  (maxxy+0.5)**2 + (minxy-0.5)**2),(minxy-0.5)) + np.nan_to_num(np.multiply((radius**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2), np.sqrt(radius**2 - (maxxy + 0.5)**2)),nan=0)
    m2 = np.multiply((radius**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2),(minxy+0.5)) + np.nan_to_num(np.multiply((radius**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2), np.sqrt(radius**2 - (maxxy - 0.5)**2)),nan=0)
    sgrid = np.multiply((radius**2*(0.5*(np.arcsin(m2/radius) - np.arcsin(m1/radius)) + 0.25*(np.sin(2*np.arcsin(m2/radius)) - np.sin(2*np.arcsin(m1/radius)))) - np.multiply((maxxy-0.5),(m2-m1)) + (m1-minxy+0.5)) ,((((radius**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & (radius**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | ((minxy==0)&(maxxy-0.5 < radius)&(maxxy+0.5>=radius)))))
    sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < radius**2)
    sgrid[crad,crad] = min(np.pi*radius**2,np.pi/2)
    if ((crad>0) and (radius > crad-0.5) and (radius**2 < (crad-0.5)**2+0.25)) :
        m1  = np.sqrt(rad**2 - (crad - 0.5)**2)
        m1n = m1/radius
        sg0 = 2*(radius**2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
        sgrid[2*crad+1,crad+1] = sg0
        sgrid[crad+1,2*crad+1] = sg0
        sgrid[crad+1,1]        = sg0
        sgrid[1,crad+1]        = sg0
        sgrid[2*crad,crad+1]   = sgrid[2*crad,crad+1] - sg0
        sgrid[crad+1,2*crad]   = sgrid[crad+1,2*crad] - sg0
        sgrid[crad+1,2]        = sgrid[crad+1,2]      - sg0
        sgrid[2,crad+1]        = sgrid[2,crad+1]      - sg0
    sgrid[crad,crad] = min(sgrid[crad,crad],1)
    h = sgrid/sgrid.sum()
    ndimage.convolve(im[:,:,0],  h, output = im[:,:,0], mode='nearest')
    ndimage.convolve(im[:,:,1],  h, output = im[:,:,1], mode='nearest')
    ndimage.convolve(im[:,:,2],  h, output = im[:,:,2], mode='nearest')
    im = Image.fromarray(im)
    return im

"""
def imblurmotion (im, level):

    # MATLAB version https://github.com/alexandrovteam/IMS_quality/blob/master/codebase/fspecialIM.m
    levels = [1, 2, 4, 6, 8]
    

    im = np.array(im)

    radius = levels[level]
    length = max(1,radius)
    half = (length-1)/2
    phi = (random.randint(0,180))/(180)*np.pi

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    xsign = np.sign(cosphi)
    linewdt = 1

    sx = np.fix(half*cosphi + linewdt*xsign - length*np.finfo(float).eps)
    sy = np.fix(half*sinphi + linewdt - length*np.finfo(float).eps)

    if sx > 0:
        end_sx = sx + 1
    else :
        end_sx = sx - 1 

    if sy >= 0:
        end_sy = sy + 1
    else :
        end_sy = sy - 1 
    
    [x,y] = np.meshgrid(np.arange(0,end_sx,xsign), np.arange(0,end_sy,1), indexing='xy')
    
    dist2line = (y*cosphi-x*sinphi) 

    rad = np.sqrt(x**2 + y**2)
    x2lastpix = half - abs((x[(rad >= half)&(abs(dist2line)<=linewdt)] + dist2line[(rad >= half)&(abs(dist2line)<=linewdt)]*sinphi)/cosphi)

    dist2line[(rad >= half)&(abs(dist2line)<=linewdt)] = np.sqrt(dist2line[(rad >= half)&(abs(dist2line)<=linewdt)]**2 + x2lastpix**2)
    dist2line = linewdt + np.finfo(float).eps - abs(dist2line)
    dist2line[dist2line<0] = 0 


    h1 = np.rot90(dist2line,2)
    h2 = np.zeros([h1.shape[0]*2-1,h1.shape[1]*2-1])
    h2[0:h1.shape[0],0:h1.shape[1]] = h1
    h2[h1.shape[0]-1:2*h1.shape[0]-1,h1.shape[1]-1:h1.shape[1]*2-1] = np.rot90(np.rot90(h1))
    h2 = h2/(h2.sum() + np.finfo(float).eps*length*length)
    if cosphi>0 :
        h2 = np.flipud(h2)
    
    ndimage.convolve(im[:,:,0],  h2, output = im[:,:,0], mode='nearest')
    ndimage.convolve(im[:,:,1],  h2, output = im[:,:,1], mode='nearest')
    ndimage.convolve(im[:,:,2],  h2, output = im[:,:,2], mode='nearest')
    im = Image.fromarray(im)
    return im
"""

def imblurmotion (im, level):

    levels = [12, 16, 20, 24, 28]

    kernel_size = levels[level]
    phi = random.choice([0,90])/(180)*np.pi
    kernel = np.zeros((kernel_size, kernel_size))

    im = np.array(im)

    if phi == 0:
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    else :
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    kernel/=kernel_size


    ndimage.convolve(im[:,:,0],  kernel, output = im[:,:,0], mode='nearest')
    ndimage.convolve(im[:,:,1],  kernel, output = im[:,:,1], mode='nearest')
    ndimage.convolve(im[:,:,2],  kernel, output = im[:,:,2], mode='nearest')

    im = Image.fromarray(im)

    return im

def imcolordiffuse(im,level):

    levels = [1, 3, 6, 8, 12]

    amount = levels[level]
    im = np.array(im)

    sigma = 1.5*amount + 2
    scaling = amount

    lab = color.rgb2lab(im)
    l = lab[:,:,0]

    lab = filters.gaussian(lab, sigma=sigma, channel_axis=-1)* scaling
    lab[:,:,0] = l
    im = 255*color.lab2rgb(lab)
    im = np.clip(im, 0, 255)
    im = Image.fromarray(np.uint8(im))
    return im


def imcolorshift(im,level):

    
    levels = [1, 3, 6, 8, 12]
    amount = levels[level]

    im = np.float32(np.array(im)/255.0)
    # RGB to Gray
    x =  0.2989 * im[:,:,0] + 0.5870 * im[:,:,1]+ 0.1140 * im[:,:,2]

    dx = np.gradient(x,axis=0)
    dy = np.gradient(x,axis=1)
    e = np.hypot(dx, dy)  # magnitude
    
    e = filters.gaussian(e, sigma=4)

    e = mapmm(e)
    e = np.clip(e,0.1,1)
    e = mapmm(e)

    percdev = [1, 1]

    valuehi = np.percentile(e,100-percdev[1])
    valuelo = 1-np.percentile(1-e,100-percdev[0])

    e = np.clip(e,valuelo,valuehi)
    e = mapmm(e)

    channel = 1
    g = im[:,:,channel]
    amt_shift = np.uint8(np.round((normalize(np.random.random([1,2]), norm='l2', axis=1) * amount)))

    padding = np.multiply(int(np.max(amt_shift)),[1, 1])

    y = np.pad(g, padding, 'symmetric')
    y = np.roll(y, amt_shift.reshape(-1))

    sl = padding[0]

    g = y [sl:-sl,sl:-sl]

    J = im
    J[:,:,channel] = np.multiply(g,e) + np.multiply(J[:,:,channel],(1-e))
    J = J * 255.0
    
    im = Image.fromarray(np.uint8(J))
    return im


def imcolorsaturate(im,level):

    
    levels = [0.4, 0.2, 0.1, 0, -0.4]
    amount = levels[level]

    im = np.array(im)
    hsvIm = color.rgb2hsv(im)
    hsvIm[:,:,1] = hsvIm[:,:,1] * amount
    im = color.hsv2rgb(hsvIm) * 255.0
    im = np.clip(im,0,255)
    im = Image.fromarray(np.uint8(im))
    
    return im



def imsaturate(im,level):

    levels = [1, 2, 3, 6, 9]
    amount = levels[level]

    lab = color.rgb2lab(im)
    lab[:,:,1:] = lab[:,:,1:] * amount
    im = color.lab2rgb(lab) * 255.0
    im = np.clip(im,0,255)
    im = Image.fromarray(np.uint8(im))
    
    return im


def imcompressjpeg(im,level):

    levels = [70, 43, 36, 24, 7]
    amount = levels[level]

    imgByteArr = io.BytesIO()
    im.save(imgByteArr, format='JPEG',quality=amount)
    im1 = Image.open(imgByteArr)

    return im1

def imnoisegauss(im, level):
    levels = [0.001, 0.002, 0.003, 0.005, 0.01]

    im = np.float32(np.array(im)/255.0)

    row,col,ch= im.shape

    var = levels[level]
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = im + gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)

    return Image.fromarray(noisy.astype('uint8'))

def imnoisecolormap(im, level):

    levels = [0.0001, 0.0005, 0.001, 0.002, 0.003]
    var = levels[level]
    mean = 0

    im = np.array(im)
    ycbcr = color.rgb2ycbcr(im)
    ycbcr = ycbcr/ 255.0

    row,col,ch= ycbcr.shape
    sigma = var**0.5
    gauss = np.array(ycbcr.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = ycbcr + gauss

    im_dist = color.ycbcr2rgb(noisy * 255.0) * 255.0
    im_dist = np.clip(im_dist,0,255)
    return Image.fromarray(im_dist.astype('uint8'))


def imnoiseimpulse(im, level):

    levels = [0.001, 0.005, 0.01, 0.02, 0.03]
    prob = levels[level]

    im = np.array(im)
    output = im

    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
    
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white


    return Image.fromarray(output.astype('uint8'))

def imnoisemultiplicative(im, level):

    levels = [0.001, 0.005, 0.01, 0.02, 0.05]

    im = np.float32(np.array(im)/255.0)

    row,col,ch= im.shape

    var = levels[level]
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = im + im * gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)

    return Image.fromarray(noisy.astype('uint8'))

def imdenoise (im,level):

    levels = [0.001, 0.002, 0.003, 0.005, 0.01]

    im = np.float32(np.array(im)/255.0)

    row,col,ch= im.shape

    var = levels[level]
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = im + gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)

    im = Image.fromarray(noisy.astype('uint8'))

    filt_type = np.random.randint(0,2)
    if filt_type == 0:
        denoised = im.filter(ImageFilter.GaussianBlur(radius = 3)) 
    elif filt_type == 1:
        denoised = im.filter(ImageFilter.BoxBlur(radius = 2)) 

    return denoised


def imbrighten(im,level) :

 
    levels = [0.1, 0.2, 0.4, 0.7, 1.1]

    amount = levels[level]
    im = np.float32(np.array(im)/255.0)

    lab = color.rgb2lab(im)
    L = lab[:,:,0]/100.0
    L_ = curvefit(L , 0.5 + 0.5*amount)
    lab[:,:,0] = L_*100.0

    J = curvefit(im, 0.5 + 0.5*amount)

    J = (2*J + np.clip(color.lab2rgb(lab),0,1) )/3.0
    J = np.clip(J * 255.0,0,255)


    return Image.fromarray(J.astype('uint8'))


def imdarken(im, level):
    levels = [0.05, 0.1, 0.2, 0.4, 0.8]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)/255.0

    ## generate curve to fit based on amount
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    y[1] = 0.5-param/2

    ## generate interpolating function and interpolate input
    cs = interpolate.UnivariateSpline(x, y, k=2)
    yy = cs(im)
    
    ## convert back to PIL image
    im_out = np.clip(yy, 0, 1)
    im_out = (im_out*255).astype(np.uint8) 
    im_out = Image.fromarray(im_out)
    return im_out


def immeanshift(im,level) :

    levels = [0.15, 0.08, 0, -0.08, -0.15]
    amount = levels[level]

    im = np.float32(np.array(im)/255.0)

    im = im + amount
    im = im * 255.0

    im = np.clip(im,0,255)
    return Image.fromarray(im.astype('uint8'))
       

def imresizedist(im,level) :

    levels = [2,3,4,8,16]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w/amount)
    scaled_h = int(h/amount)

    resized_image = im.resize((scaled_w,scaled_h))

    im = resized_image.resize((w,h))
    return im

def imresizedist_bilinear(im,level) :

    levels = [2,3,4,8,16]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w/amount)
    scaled_h = int(h/amount)

    resized_image = im.resize((scaled_w,scaled_h),Image.BILINEAR)

    im = resized_image.resize((w,h),Image.BILINEAR)
    return im


def imresizedist_nearest(im,level) :

    levels = [2,3,4,5,6]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w/amount)
    scaled_h = int(h/amount)

    resized_image = im.resize((scaled_w,scaled_h),Image.NEAREST)

    im = resized_image.resize((w,h),Image.NEAREST)
    return im

def imresizedist_lanczos(im,level) :

    levels = [2,3,4,8,16]
    amount = levels[level]

    size = im.size

    w = size[0]
    h = size[1]

    scaled_w = int(w/amount)
    scaled_h = int(h/amount)

    resized_image = im.resize((scaled_w,scaled_h),Image.LANCZOS)

    im = resized_image.resize((w,h),Image.LANCZOS)
    return im


def imsharpenHi(im, level):
    levels = [1, 2, 3, 6, 12]
    param = levels[level]
    ## param range to be use -> double from matlab
    ## convert PIL-RGB to LAB for operation in L space
    im = np.array(im).astype(np.float32)
    LAB = color.rgb2lab(im)
    im_L = LAB[:,:,0]

    ## compute laplacians
    gy = np.gradient(im_L, axis=0)
    gx = np.gradient(im_L, axis=1)
    ggy = np.gradient(gy, axis=0)
    ggx = np.gradient(gx, axis=1)
    laplacian = ggx + ggy

    ## subtract blurred version from image to sharpen
    im_out = im_L - param*laplacian

    ## clip L space in 0-100
    im_out = np.clip(im_out, 0, 100)

    ## convert LAB to PIL-RGB 
    LAB[:,:,0] = im_out
    im_out = 255*color.lab2rgb(LAB)
    im_out = im_out.astype(np.uint8)
    im_out = Image.fromarray(im_out)
    return im_out

def imcontrastc(im, level):
    levels = [0.3, 0.15, 0, -0.4, -0.6]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)/255.0

    ## generate curve to fit based on amount->param
    coef = [[0.3, 0.5, 0.7],[0.25-param/4, 0.5, 0.75+param/4]]
    defa = 0
    x = [0, 0, 0, 0, 1]
    x[1:-1] = coef[0] 
    y = [0, 0, 0, 0, 1]
    y[1:-1] = coef[1] 

    ## generate interpolating function and interpolate input
    cs = interpolate.UnivariateSpline(x, y)
    yy = cs(im)
    
    ## convert back to PIL image
    im_out = np.clip(yy, 0, 1)
    im_out = (im_out*255).astype(np.uint8) 
    im_out = Image.fromarray(im_out)
    return im_out

def imcolorblock(im, level):
    levels = [2, 4, 6, 8, 10]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)
    h, w, _ = im.shape

    ## define patchsize
    patch_size = [32, 32]

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    block = np.ones((patch_size[0], patch_size[1], 3))

    ## place the color blocks at random
    for i in range(0, param):
        color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        x = int(random.uniform(0, 1) * w_max)
        y = int(random.uniform(0, 1) * h_max)
        im[y:y+patch_size[0], x:x+patch_size[1],:] = color*block
    
    ## convert back to PIL image
    im_out = (im).astype(np.uint8) 
    im_out = Image.fromarray(im_out)
    return im_out



def impixelate(im, level):
    levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    param = levels[level]
    z = 0.95 - param**(0.6)
    size_z_1 = int(z*im.width)
    size_z_2 = int(z*im.height)

    im_out = im.resize((size_z_1, size_z_2), resample=PIL.Image.NEAREST)
    im_out = im_out.resize((im.width,im.height), resample=PIL.Image.NEAREST)
    return im_out

def imnoneccentricity(im, level):
    levels = [20, 40, 60, 80, 100]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)
    h, w, _ = im.shape

    ## define patchsize
    patch_size = [16, 16]

    radius = 16

    h_min = radius
    w_min = radius

    h_max = h - patch_size[0] - radius
    w_max = w - patch_size[1] - radius

    block = np.ones((patch_size[0], patch_size[1], 3))

    ## place the color blocks at random
    for i in range(0, param):
        w_start = int(random.uniform(0, 1) * (w_max - w_min)) + w_min
        h_start = int(random.uniform(0, 1) * (h_max - h_min)) + h_min
        patch = im[h_start:h_start+patch_size[0],w_start:w_start+patch_size[1],:]
        
        rand_w_start = int((random.uniform(0, 1) - 0.5)*radius + w_start)
        rand_h_start = int((random.uniform(0, 1) - 0.5)*radius + h_start)
        im[rand_h_start:rand_h_start+patch_size[0],rand_w_start:rand_w_start+patch_size[1],:] = patch
    
    ## convert back to PIL image
    im_out = (im).astype(np.uint8) 
    im_out = Image.fromarray(im_out)
    return im_out

def imwarpmap(im, shifts):
    sy, sx = shifts[:,:,0], shifts[:,:,1] 
    ## create mesh-grid for image shape
    [xx, yy] = np.meshgrid(range(0,shifts.shape[1]), range(0,shifts.shape[0]))

    ## check whether grey image or RGB
    shape = im.shape
    im_out = im
    if len(shape)>2:
        ch = shape[-1]
    else:
        ch = 1

    ## iterate function over each channel
    for i in range(ch):
        im_out[:,:,i] = ndimage.map_coordinates(im[:,:,i], [(yy-sy).ravel(), (xx-sx).ravel()], order = 3, mode = 'reflect').reshape(im.shape[:2])

    ## clip image between 0-255
    im_out = np.clip(im_out, 0, 255)

    return im_out

def imjitter(im, level):
    levels = [0.05, 0.1, 0.2, 0.5, 1]
    param = levels[level]
    ## convert 0-1
    im = np.array(im).astype(np.float32)
    h, w, _ = im.shape

    sz = [h,w,2]

    ## iterate image-warp over for 5 times
    J = im
    for i in range(0,5):
        ## generate random shift map
        shifts = np.random.randn(h,w,2)*param
        J = imwarpmap(J, shifts)

    ## convert back to PIL image
    im_out = (J).astype(np.uint8) 
    im_out = Image.fromarray(im_out)
    return im_out

"""
im = Image.open("IMG_1651.png")
for level in dist_level:
    
    im_dist = imcompressjpeg(im,level)
    im_dist.save("level"+str(level)+".png")

"""
