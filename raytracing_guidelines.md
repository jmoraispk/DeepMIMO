
General: 

- always isotropic antennas
- 2 pol > 1 pol
- diffuse scattering 
- 


In Wireless Insite:

- The only 1 required outputs: paths
- For robustness: Use a single study area, antenna and waveform.
- Define as many <points> and <grids> as needed. 
- When defining a set of <points>, define a single location per set. 
- Ensure the txrx elements in your workspace have growing ids ("1,2,3.." and not "1 3 2")
- Keep the default VSWR of 1 (perfect antenna matching).


Left to test:
- When the reference of a grid is set to terrain, do the points heights change
  according to the terray underneath them? Or only according to the terrain
  on the corner of the grid? Hypothesis: the 2nd. 
  Limitation: our code considers the 2nd option and won't locate points properly.
- 