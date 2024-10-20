import imageio
import numpy as np
import matplotlib.pyplot as plt
#from scipy_filter import butterworth_lowpass, lfilter
from typing import Tuple


def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = np.array([])
    m = np.arange(-N + 1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1
    return z, p, k


def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo ** degree

    return z_lp, p_lp, k_lp


def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw / 2
    p_lp = p * bw / 2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = np.concatenate((z_lp + np.sqrt(z_lp ** 2 - wo ** 2),
                           z_lp - np.sqrt(z_lp ** 2 - wo ** 2)))
    p_bp = np.concatenate((p_lp + np.sqrt(p_lp ** 2 - wo ** 2),
                           p_lp - np.sqrt(p_lp ** 2 - wo ** 2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = np.append(z_bp, np.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw ** degree

    return z_bp, p_bp, k_bp


def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    fs = float(fs)

    degree = _relative_degree(z, p)

    fs2 = 2.0 * fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_z, p_z, k_z


def zpk2tf(z, p, k):
    r"""
    Return polynomial transfer function representation from zeros and poles

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.
    """
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    if len(z.shape) > 1:
        temp = np.poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * np.poly(z[i])
    else:
        b = k * np.poly(z)
    a = np.atleast_1d(np.poly(p))

    # Use real output if possible. Copied from np.poly, since
    # we can't depend on a specific version of np.
    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                      np.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a


def butterworth_lowpass(N: int, Wn: float) -> Tuple[np.ndarray, np.ndarray]:
    """LP Butterworth digital filter

    Args:
        N (int): order
        Wn (float): cutoff frequency

    Returns:
        b : ndarray
            Numerator polynomial coefficients.
        a : ndarray
            Denominator polynomial coefficients.
    """
    assert np.size(Wn) == 1, "Must specify a single critical frequency Wn for lowpass or highpass filter"
    assert np.all(Wn > 0) and np.all(Wn < 1), "Digital filter critical frequencies must be 0 < Wn < 1"

    z, p, k = buttap(N)
    warped = 4 * np.tan(np.pi * Wn / 2)  # digital
    z, p, k = lp2lp_zpk(z, p, k, wo=warped)
    z, p, k = bilinear_zpk(z, p, k, fs=2)
    b, a = zpk2tf(z, p, k)

    return b, a


def butterworth_bandpass(N: int, Wn: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """LP Butterworth digital filter

    Args:
        N (int): order
        Wn (Tuple[float, float]): band pass frequencies

    Returns:
        b : ndarray
            Numerator polynomial coefficients.
        a : ndarray
            Denominator polynomial coefficients.
    """
    Wn = np.array(Wn)
    assert np.size(Wn) == 2, "Must specify a single critical frequency Wn for lowpass or highpass filter"
    assert np.all(Wn > 0) and np.all(Wn < 1), "Digital filter critical frequencies must be 0 < Wn < 1"

    z, p, k = buttap(N)
    warped = 4 * np.tan(np.pi * Wn / 2)  # digital

    bw = warped[1] - warped[0]
    wo = np.sqrt(warped[0] * warped[1])
    z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
    z, p, k = bilinear_zpk(z, p, k, fs=2)
    b, a = zpk2tf(z, p, k)

    return b, a


def lfilter(b, a, x):
    """A simple implementation of a 1-D linear filter."""
    a = np.array(a)
    b = np.array(b)
    y = np.zeros_like(x)
    a0 = a[0]
    if a0 != 1:
        a = a / a0
        b = b / a0
    for i in range(len(x)):
        for j in range(len(b)):
            if i - j >= 0:
                y[i] += b[j] * x[i - j]
        for j in range(1, len(a)):
            if i - j >= 0:
                y[i] -= a[j] * y[i - j]
    return y


if __name__ == '__main__':
    video_path = r'C:\Users\mm200507d\Desktop\psiml2\public\public\set\0\video.mp4'
    sound_path = r'C:\Users\mm200507d\Desktop\psiml2\public\public\set\0\sound.npy'

    video = imageio.get_reader(video_path)

    sound = np.load(sound_path)

    first_frame = video.get_data(0)
    video.close()
    #plt.figure()
    #plt.imshow(first_frame)
    #plt.show()

    #print(first_frame.shape)

    gray_im=0.3*first_frame[:,:,0]+0.59*first_frame[:,:,1]+0.11*first_frame[:,:,2]

    bin_im=gray_im>=60
    #print(bin_im)

    '''plt.figure()
    plt.imshow(gray_im,cmap='gray')
    

    plt.figure()
    plt.imshow(bin_im, cmap='gray')
    video.close()
    plt.show()
    '''

    '''
    left_point=[-1,-1]
    flag=0
    for j in range (bin_im.shape[1]):
        if flag: break
        for i in range (bin_im.shape[0]):
            if bin_im[i,j] and not bin_im[i+10,j]:
                left_point=[i,j]
                flag=1
                break


    #print(curr)
    best=[-1,-1]
    for k in range(5):
        curr = left_point.copy()
        curr[0]=left_point[0]+k
        while bin_im[curr[0],curr[1]]:
            curr[1]+=1
        if curr[1]>best[1]:
            best=curr

    #print(curr)
    center=[best[0],int(np.round((left_point[1]+best[1])/2.0))]
    #print(center)

    print(center[0],end=" ")
    print(center[1])
    '''

    er_mask=np.ones([7,7],dtype='bool')

    padded_im=np.pad(bin_im,((3, 3), (3, 3)), mode='constant', constant_values=False)

    #plt.figure()
    #plt.imshow(padded_im, cmap='gray')

    #print(bin_im.shape)
    #print(padded_im.shape)


    er_im=np.zeros(padded_im.shape,dtype='bool')
    for i in range (bin_im.shape[0]):
        for j in range(bin_im.shape[1]):
            #print(er_im[i:i+7,j:j+7]&er_mask)
            er_im[i+3,j+3]=np.all((padded_im[i:i+7,j:j+7]&er_mask))

    #print(padded_im)
    #plt.figure()
    #plt.imshow(er_im,cmap='gray')

    #print(er_im.shape)
    #print(er_im)

    dil_im = np.zeros(padded_im.shape, dtype='bool')
    for i in range(bin_im.shape[0]):
        for j in range(bin_im.shape[1]):
            # print(er_im[i:i+7,j:j+7]&er_mask)
            dil_im[i + 3, j + 3] = np.any((er_im[i:i + 7, j:j + 7] & er_mask))

    dil_im=dil_im[3:dil_im.shape[0]-3,3:dil_im.shape[1]-3]

    left_point = [-1, -1]
    flag = 0
    for j in range(bin_im.shape[1]):
        if flag: break
        for i in range(bin_im.shape[0]):
            if dil_im[i, j]:
                left_point = [i, j]
                flag = 1
                break

    # print(curr)
    best = [-1, -1]
    for k in range(5):
        curr = left_point.copy()
        curr[0] = left_point[0] + k
        while dil_im[curr[0], curr[1]]:
            curr[1] += 1
        if curr[1] >= best[1]:
            best = curr

    # print(curr)
    center = [best[0], int(np.round((left_point[1] + best[1]) / 2.0))]
    # print(center)

    print(center[0], end=" ")
    print(center[1])


    new_im=dil_im^bin_im

    new_im_padded=np.pad(new_im,((1, 1), (1, 1)), mode='constant', constant_values=False)
    new_im2=np.zeros(new_im_padded.shape,dtype='bool')
    for i in range(bin_im.shape[0]):
        for j in range(bin_im.shape[1]):
            new_im2[i+1,j+1]=np.median(new_im_padded[i:i+3,j:j+3])

    new_im2=new_im2[1:new_im2.shape[0]-1,1:new_im2.shape[1]-1]

    #print(bin_im.shape)
    #print(new_im2.shape)

    ''''
    plt.figure()
    plt.imshow(dil_im, cmap='gray')
    plt.show()

    

    plt.figure()
    plt.imshow(new_im, cmap='gray')

    plt.figure()
    plt.imshow(new_im2, cmap='gray')
    plt.show()
    '''

    left_point2 = [-1, -1]
    flag2 = 0
    for j in range(new_im2.shape[1]):
        if flag2: break
        for i in range(bin_im.shape[0]):
            if new_im2[i, j] :
                left_point2 = [i, j]
                flag2 = 1
                break


    right_best=[-1,-1]
    for k in range(3):
        right_point2 = left_point2.copy()
        right_point2[0]=left_point2[0]+k
        while new_im2[right_point2[0],right_point2[1]]:
            right_point2[1]+=1
        if right_point2[1]>right_best[1]:
            right_best=right_point2

    down_best = [-1, -1]
    for k in range(3):
        down_point2=left_point2.copy()
        down_point2[1]=left_point2[1]+k
        while new_im2[down_point2[0],down_point2[1]]:
            down_point2[0]+=1
        if down_point2[0]>down_best[0]:
            down_best=down_point2

    center_col=int(np.round((left_point2[1]+right_best[1])/2.0))
    center_row=int(np.round((left_point2[0]+down_best[0])/2.0))

    print(center_row,end=' ')
    print(center_col)

    fs=44100.0
    t=np.arange(len(sound)) / fs

    #plt.figure()
    #plt.plot(t,sound)

    order=4
    cutoff_freq=4000

    b,a=butterworth_lowpass(order,cutoff_freq/(fs/2))

    sound_filt=lfilter(b,a,sound)

    #plt.figure()
    #plt.plot(t,sound_filt)
    #plt.show()

    duration=15/1000;
    samples=fs*duration
    #print(samples)
    T=0.5
    peaks = []
    for i in range(1, len(sound_filt) - 1):
        if sound_filt[i] > T and sound_filt[i]>sound_filt[i-1] and sound_filt[i]>sound_filt[i+1]:
            peaks.append(i)

    real_peaks=[peaks[0]]
    for i in range(1,len(peaks)):
        if peaks[i]-peaks[i-1]>samples:
            real_peaks.append(peaks[i])

    #print(len(peaks))
    #print(peaks)

    #print(real_peaks)
    print(len(real_peaks))