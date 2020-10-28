# Fourier Transforms and Applications in Seismology:

When earth material properties are constant in any of the cartesian variables (t,x,y,z) then it is useful to Fourier transform (FT) that variable.
In seismology, the earth does not change with time (the ocean does!) so for the earth, we can generally gain by Fourier transforming the time axis thereby converting time-dependent differential equations (hard) to algebraic equations (easier) in frequency (temporal frequency).

In seismology, the earth generally changes rather strongly with depth, so we cannot usefully Fourier transform the depth z axis and we are stuck with differential equations in z. On the other hand, we can model a layered earth where each layer has material properties that are constant in z. Then we get analytic solutions in layers and we need to patch them together.

Thirty years ago, computers were so weak that we always Fourier transformed the x and y coordinates. That meant that their analyses were limited to earth models in which velocity was horizontally layered. Today we still often Fourier transform t,x,y but not z, so we reduce the partial differential equations of physics to ordinary differential equations (ODEs). A big advantage of knowing FT theory is that it enables us to visualize physical behavior without us needing to use a computer.

The Fourier transform variables are called frequencies. For each axis (t,x,y,z) we have a corresponding frequency $(\omega,k_x,k_y,k_z)$.The k's are spatial frequencies, $\omega$ is the temporal frequency.

The frequency is inverse to the wavelength. Question: A seismic wave from the fast earth goes into the slow ocean. The temporal frequency stays the same. What happens to the spatial frequency (inverse spatial wavelength)?

In a layered earth, the horizonal spatial frequency is a constant function of depth. We will find this to be Snell's law.





# Interactive Python Notebook:
The following python notebook has been designed to offer an interactive and a visual approach to forming a general, mathematical insight about Fourier Series and Transforms:
[Fourier Series and Transforms - An Interactive, Visual Approach .ipynb]()


# Learning Resources:


For further gaining insight and building general understanding, the following are good start-points for learning Fourier Series and Transforms:

**<ins>Textbook Library:</ins>**
[Textbooks Drive Folder]()

**<ins>Videos:</ins>**
- **Explanations/Insights on Fourier Series and Transforms:**

  - [But what is the Fourier Transform? A visual introduction.](https://www.youtube.com/watch?v=spUNpyF58BY&t=11s)

  - [But what is a Fourier series? From heat flow to circle drawings | DE4](https://www.youtube.com/watch?v=r6sGWTCMz2k&list=PLide-NR5bCoFg5lFQbnOt5fvlrpnpdcrX&index=1&t=14s)

  - [The intuition behind Fourier and Laplace transforms I was never taught in school](https://www.youtube.com/watch?v=3gjJDuCAEQQ)

  - [Compute Fourier Series Representation of a Function](https://www.youtube.com/watch?v=SnzSpbQ2mcQ&t=128s)

  - [Pure Fourier series animation montage](https://www.youtube.com/watch?v=-qgreAUpPwM)


- **Applications in Physics:**
  - [The more general uncertainty principle, beyond quantum](https://www.youtube.com/watch?v=MBnnXbOM5S4)

  - [Solving the heat equation | DE3](https://www.youtube.com/watch?v=ToIXSwZ1pJU&t=14s)


**<ins>Papers:</ins>**
- [Fourier Analysis Made Easy Part I](http://complextoreal.com/wp-content/uploads/2012/12/fft1.pdf)

- [Fourier Analysis Made Easy Part II](http://www.mbfys.ru.nl/~robvdw/DGCN22/PRACTICUM_2011/MATLAB_FFT/fft2x.pdf)

 
**<ins>Webpages:</ins>**
- [An Interactive Introduction to Fourier Transforms](http://www.jezzamon.com/fourier/index.html)

- [A Tale of Math & Art: Creating the Fourier Series Harmonic Circles Visualization](https://alex.miller.im/posts/fourier-series-spinning-circles-visualization/)

- [An Interactive Guide To The Fourier Transform](https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/)

- [Theoretical Significance of Fourier Analysis](https://www.reddit.com/r/math/comments/6lt659/theoretical_significance_of_fourier_analysis/)
