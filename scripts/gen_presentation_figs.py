"""
This will generate all plots used in my PyData2017 talk in Amsterdam. Please use the powerpoint presentation for 
more info in the plots. 

Title: 'Smoothing your data with polynomial fitting: a signal processing perspective'
Author: Cees Taal
email: chtaal@gmail.com
"""

## imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import smoothing as sm
import scipy.signal as sig

## general settings
dpi = 300
rc('text', usetex=True) # turn on latex interpreter ( if this doesn't work leave it out and get more ugly plots :) )

## slide 4
y = np.ndfromtxt('data\\temps.txt')
winl = 101

y_fit = sm.polyfit_window(y, window_length=winl, deg=3)
y_fit = np.roll(y_fit, -int(winl/2+1))  # compensate for delay

t = np.arange(len(y))*3  # adjust for sample-rate which was 1/3 hertz

plt.figure(figsize=(5, 3))
plt.plot(t, y, label='noisy')
plt.xlabel('Time (s)')
plt.ylabel('Temperate (C)')
plt.grid(True)
plt.xlim([500, 3500])
plt.ylim([22.2, 23])
plt.legend()
plt.tight_layout()
plt.gcf().savefig('figs\\noisy_signal.png', dpi=dpi)

plt.figure(figsize=(5, 3))
plt.plot(t, y, label='noisy')
plt.plot(t, y_fit, linewidth=2, label='enhanced')
plt.xlabel('Time (s)')
plt.ylabel('Temperate (C)')
plt.grid(True)
plt.xlim([500, 3500])
plt.ylim([22.2, 23])
plt.legend()
plt.tight_layout()
plt.gcf().savefig('figs\\noisy_signal_enhanced.png', dpi=dpi)

## slide 5
x = [1, 2, 3, 4, 5]
y = [3, 2, 0, 4, 5]

fig = plt.figure(figsize=(5, 4))

plt.plot(x, y, 'o', label='$y$')
fig.savefig('figs\\polyfit_in.png', dpi=300)
plt.ylim([-6, 7])
plt.xlim([0, 6])
plt.grid(True)
plt.xlabel('$x$')
plt.legend(loc=8)
fig.savefig('figs\\polyfit_in.png', dpi=300)
n = 5

for deg in [1, 2, 4]:
    p = np.polyfit(x, y, deg=deg)
    x_fit = np.linspace(0, 6, 100)
    y_fit = np.polyval(p, x_fit)

    plt.plot(x_fit, y_fit, label='$f(x), m=' + str(deg) + '$')

    plt.ylim([-6, 7])
    plt.xlim([0, 6])
    plt.grid(True)
    plt.xlabel('$x$')
    plt.legend(loc=8)

    fig.savefig('figs\\polyfit' + str(deg) + '.png', dpi=dpi)

## slide6
deg = 3

# first generate some noisy data
p_clean = np.polyfit([1, 2, 3, 4, 5], [3, 2, 0, 4, 5], deg=deg)
x = np.linspace(0, 6, 100)
y_clean = np.polyval(p_clean, x)
y_noisy = y_clean + np.random.randn(len(x))

# now fit to show
p_noisy = np.polyfit(x, y_noisy, deg=deg)
y_fit = np.polyval(p_noisy, x)

plt.plot(x, y_clean, label='$y$')
plt.plot(x, y_noisy, '.', label='$y+\epsilon$')
plt.plot(x, y_fit, label='$f(y+\epsilon)$')

plt.grid(True)
plt.xlabel('$x$')
plt.legend(loc='upper center')
plt.xlim([0, 6])

plt.gcf().savefig('figs\\polyfit_noisy.png', dpi=dpi)

## slide 7
y = np.ndfromtxt('data\\temps.txt')
winl = 101

x = np.arange(len(y))
p = np.polyfit(x, y, deg=10)
y_fit = np.polyval(p, x)

t = np.arange(len(y))*3  # adjust for sample-rate

plt.plot(t, y)
plt.plot(t, y_fit, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperate (C)')
plt.grid()
plt.xlim([500, 3500])
plt.ylim([22.2, 23])

plt.gcf().savefig('figs\\long_signal.png', dpi=dpi)

## slide 8
y = np.ndfromtxt('data\\temps.txt')
win_length = 101
center = int(win_length/2 + 1)
x = np.arange(win_length)

offsets = np.arange(60, 160, 15)

x_center = np.zeros(len(offsets))
y_center = np.zeros(len(offsets))

for cnt, offset in enumerate(offsets):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y))*3, y)
    p = np.polyfit(x, y[offset:(offset+win_length)], deg=4)
    y_fit = np.polyval(p, x)

    x_center[cnt] = x[center]+offset
    y_center[cnt] = y_fit[center]

    plt.plot((x+offset)*3, y_fit, linewidth=3)
    plt.plot(x_center[:cnt+1]*3, y_center[:cnt+1], 'o-', markersize=10)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperate (C)')
    plt.grid()

    plt.xlim([150, 900])
    plt.ylim([22.2, 22.6])

    fig.savefig('figs\\polyfit_win' + str(offset) + '.png', dpi=dpi)

## slide 9
y = np.ndfromtxt('data\\temps.txt')
y_fit = sm.polyfit_window(y, window_length=winl, deg=3)
y_fit = np.roll(y_fit, -int(winl/2+1))  # compensate for delay

t = np.arange(len(y))*3  # adjust for sample-rate

fig = plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.plot(t, y_fit, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperate (C)')
plt.grid()
plt.xlim([500, 3500])
plt.ylim([22.2, 23])

plt.gcf().savefig('figs\\long_signal_windowed.png', dpi=dpi)

## slide 12
win_length = 160
fft_length = win_length*4
t = np.arange(win_length)/win_length
omega1 = 2*np.pi*2
omega2 = 2*np.pi*20

x = 0.2*np.cos(omega1*t) + np.cos(omega2*t)
X = 2*np.abs(np.fft.rfft(x, fft_length))/win_length

plt.figure(figsize=(5, 4))
plt.plot(t, x)
plt.grid(True)
plt.xlabel('Time(s')
plt.tight_layout()
plt.gcf().savefig('figs\\fourier_ex_time.png', dpi=dpi)

plt.figure(figsize=(5, 4))
f = np.fft.rfftfreq(fft_length, d=1/win_length)
plt.plot(f, X)
plt.grid(True)
plt.xlabel('Frequency(Hz)')
plt.tight_layout()
plt.gcf().savefig('figs\\fourier_ex_freq.png', dpi=dpi)

## slide 15
winl = 101
pos = np.round((winl-1)/2)
t = np.arange(winl)/winl

for polyorder in [1, 3, 5, 7]:
    h = sig.savgol_coeffs(window_length=winl, polyorder=polyorder, pos=pos)
    plt.plot(h, label='$h, m=$' + str(polyorder))

plt.xlabel('$n$')
plt.grid(True)
plt.legend()
plt.gcf().savefig('figs\\sg_impulse', dpi=dpi)

## slide 16
fs = 160
win_length = fs*2
fft_length = win_length*2
t = np.arange(win_length)/fs
omega1 = 2*np.pi*2
omega2 = 2*np.pi*20

x = 0.2*np.cos(omega1*t) + np.cos(omega2*t)
h = sig.savgol_coeffs(window_length=85, polyorder=8)
y = sig.convolve(x, h)
y = y[((85-1)/2):]

f = np.fft.rfftfreq(fft_length, d=1/fs)
X = 2*np.abs(np.fft.rfft(x, fft_length))/win_length
H = np.abs(np.fft.rfft(h, fft_length))
Y = 2*np.abs(np.fft.rfft(y, fft_length))/win_length

plt.figure(figsize=(12, 4))

ax = plt.subplot(231)
plt.plot(t, x)
plt.grid(True)
plt.xlabel('Time(s)')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.plot(np.arange(len(h))/fs, h)
plt.grid(True)
plt.xlabel('Time(s)')
plt.xticks([])
plt.yticks([])

plt.subplot(233, sharex=ax, sharey=ax)
plt.plot(np.arange(len(y))/fs, y)
plt.grid(True)
plt.xlabel('Time(s)')
plt.xlim([0, 2])
plt.xticks([])
plt.yticks([])

plt.subplot(234)
plt.plot(f, X, linewidth=2)
plt.xlim([0, 40])
plt.ylim([0, 1])
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.plot(f, H)
plt.xlim([0, 40])
plt.ylim([0, 1])
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.xticks([])
plt.yticks([])

plt.subplot(236)
plt.plot(f, Y)
plt.xlim([0, 40])
plt.ylim([0, 1])
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.xticks([])
plt.yticks([])

plt.gcf().savefig('figs\\conv_theorem', dpi=dpi)

## slide 17
winl = 15
fft_length = winl*32
pos = np.round((winl-1)/2)
f = np.fft.rfftfreq(fft_length, d=1/winl)

plt.figure()

for polyorder in [2, 6, 10]:
    h = sig.savgol_coeffs(window_length=winl, polyorder=polyorder, pos=pos)
    f = np.fft.rfftfreq(fft_length, d=1 / fs)
    H = np.abs(np.fft.rfft(h, fft_length))
    plt.plot(f, H, label='$m=' + str(polyorder) + '$')

plt.xlabel('Frequency (Hz)')
plt.xlim([0, 80])
plt.grid(True)
plt.legend()
plt.gcf().savefig('figs\\sg_freq_degree.png', dpi=dpi)

## slide 18
winl = 81
fft_length = winl*32
f = np.fft.rfftfreq(fft_length, d=1/winl)

plt.figure()

for winl in [27, 15, 9]:
    h = sig.savgol_coeffs(window_length=winl, polyorder=6)
    f = np.fft.rfftfreq(fft_length, d=1 / fs)
    H = np.abs(np.fft.rfft(h, fft_length))
    plt.plot(f, H, label='$N=' + str(winl) + '$')

plt.xlabel('Frequency (Hz)')
plt.xlim([0, 80])
plt.grid(True)
plt.legend()
plt.gcf().savefig('figs\\sg_freq_win.png', dpi=dpi)

## slide 21/23
x = np.ndfromtxt('data\\meter.txt')

sample_rate = 800                              # sampling rate (Hz)
delta = 1/sample_rate                                               # sample period (seconds)
window_length = 201                                                  # window length (samples), should be odd!
polyorder = 5                                                      # order of the fitted polynomial
deriv = 0                                                          # derative order of the polynial
pos = int((window_length-1)/2)                                      # sample position used for polynomial evaluation
b = sig.savgol_coeffs(window_length, polyorder, deriv, delta, pos)
y = sig.convolve(x-np.mean(x), b)[:len(x)]
t = np.arange(len(y))/sample_rate

plt.figure(figsize=(12, 4))
plt.plot(t, x-np.mean(x), label='unfiltered')
plt.grid()
plt.xlabel('Time(s)')
plt.legend()
plt.gcf().savefig('figs\\meter_noisy.png', dpi=dpi)

plt.figure(figsize=(12, 4))
plt.plot(t, x-np.mean(x), label='unfiltered')
plt.plot(t, y, label='SG-filtered')
plt.grid()
plt.xlabel('Time(s)')
plt.legend()
plt.gcf().savefig('figs\\meter_enhanced.png', dpi=dpi)


## slide 22
win_size = 8000
fft_size = win_size*8
X = np.mean(np.abs(sm.st_rfft(x-np.mean(x), win_size, win_size/8, fft_size))**2, axis=1)
Y = np.mean(np.abs(sm.st_rfft(y-np.mean(y), win_size, win_size/2, fft_size))**2, axis=1)

f = np.fft.rfftfreq(fft_size, d=1/sample_rate)

plt.figure(figsize=(12, 4))
plt.plot(f, 10*np.log10(X), label='unfiltered')
plt.xlim([0, 60])
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('level (dB)')
plt.ylim([15, 70])
plt.legend()
plt.tight_layout()
plt.gcf().savefig('figs\\meter_f_noisy.png', dpi=dpi)


plt.figure(figsize=(12, 4))
plt.plot(f, 10*np.log10(X), label='unfiltered')
fh, H = np.abs(sig.filter_design.freqz(b, worN=1000))

H *= 45
H += 15

plt.plot(fh*sample_rate/(2*np.pi), np.abs(H), '--', label='SG-filter $(N=201, m=5)$')
plt.xlim([0, 60])
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('level (dB)')
plt.ylim([15, 70])
plt.legend()
plt.tight_layout()
plt.gcf().savefig('figs\\meter_f_noisy_sg.png', dpi=dpi)


plt.figure(figsize=(12, 4))
plt.plot(f, 10*np.log10(Y), label='SG-filtered')
plt.xlim([0, 60])
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('level (dB)')
plt.ylim([15, 70])
plt.legend()
plt.tight_layout()
plt.gcf().savefig('figs\\meter_f_enhanced.png', dpi=dpi)