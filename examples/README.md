# Examples

* [Simple](#Simple)
* [Drawing multiple pairs of particles in 1D](#Multiple1D)
* [Varying the std. dev. for the Gaussian](#VaryingStdDev)
* [Drawing pairs in 2D](#Drawing2D)
* [Multiple species](#MultipleSpecies)
* [Adding/removing/moving particles](#AddingRemoving)

## <a id="Simple"></a>Simple

A simple example is `simple.py` that shows how to sample pairs of particles.

The main steps are:

1. Create particle positions. This should be a numpy array of dimensions `n x d`, where `n` is the number of particles and `d` is the dimension of each point (all dimensions 1D, 2D, 3D or higher are possible).

2. Create the probability calculator `ProbCalculator` object. This requires the particle positions. This takes as arguments:
* `posns` - a numpy array of positions
* `dim` - integer specifiying the dimensionality of the points
* `std_dev` - the standard deviation of the Gaussian.
* `std_dev_clip_mult` - a multiplier for cutting off the probabilities. For a given particle, particles that are further than `std_dev_clip_mult * std_dev` away are not considered for draws.
If this value is too low, only particles very close to each-other are considered for pairs, and the method is inaccurate (or fails). If this value is too high, the sampling is very inefficient, especially using rejection sampling since many particles far away from one another are considered as candidates.

This value can also be excluded by specifying `std_dev_clip_mult=None`.

A good value here is e.g. `std_dev_clip_mult = 3`, such that `99.73%` of the full distribution is used.

3. Create the sampler `Sampler` object. This takes the `ProbCalculator` from the previous step.

4. Sample using:
  * Rejection sampling as follows:
  ```
  success = sampler.rejection_sample(no_tries_max=no_tries_max)
  ```
  where `success` will be a Boolean, and `no_tries_max` is the number of tries before quitting.

  * Computing the CDF (i.e. through `numpy.random.choice` with weighted probabilities) as follows:
  ```
  success = sampler.cdf_sample()
  ```

## <a id="Multiple1D"></a>Drawing multiple pairs of particles in 1D

Drawing multiple pairs of particles in 1D space is shown in the `multiple_1d.py` file.

Histogram of 100 particle positions drawn from a uniform distribution:

<img src="figures/multiple_1d_particles.png" width="300">

Samples of 1000 pairs of particles (first particle on x axis; second particle on y axis; every point is a sampled pair). Naturally, it is symmetric about `y=x`.

<img src="figures/multiple_1d_samples.png" width="300">

## <a id="VaryingStdDev"></a>Varying the std. dev. for the Gaussian

The effect of varying the std. dev. for the Gaussian is shown in the `histograms_1d.py` file.

**Note** Varying the standard deviations or the clip cutoff multiplier does **not** recalculate the distances - this would be inefficient. These are stored separately; hence changing these parameters simply applies the exponential function to all elements.

Histogram of 1000 particle positions drawn from a uniform distribution:

<img src="figures/histogram_1d_particles.png" width="300">

10000 draws for varying std. devs. (first particle on x axis; second particle on y axis; colors indictae number of draws). Naturally, it is symmetric about `y=x`.

<img src="figures/histogram_1d_1.png" width="300">
<img src="figures/histogram_1d_2.png" width="300">
<img src="figures/histogram_1d_3.png" width="300">

## <a id="Drawing2D"></a>Drawing pairs in 2D

The module supports points in any spatial dimension, although samples are more difficult to visualize. The file `sample_2d.py` illustrates this in 2D.

1000 points in 2D space drawn from a Gaussian about 0, with colors indicating the the number of times a particle is included in a sampled pair, evaluated by performing 10000 draws.

<img src="figures/sample_2d_counts.png" width="300">

For each particle, the average distance to the neighbor it was drawn with. At the outer edge, particles are not drawn because the chosen std. dev. (=1.0) is too small (for these particle, no other particle is close enough to be drawn). At the center, where many draws occur (see previous figure), the length scale is related to the chosen std. dev. of the Gaussian. As we move closer to the edge, the average distance between sampled pairs increases (although the count of draws for these particles decreases as shown in the previous figure) because particles become more sparse.

<img src="figures/sample_2d_ave_dist.png" width="300">

## <a id="MultipleSpecies"></a>Multiple species

Drawing two particles of the **same species** from a collection of multiple species is also supported.

A simple example is `simple_multispecies.py`, which should be the self-explanatory generalization of `simple.py`.

A further example is `multiple_1d_multispecies.py`, where two populations `A`,`B` exist.

<img src="figures/multiple_1d_multispecies_A.png" width="300">
<img src="figures/multiple_1d_multispecies_B.png" width="300">
<img src="figures/multiple_1d_multispecies.png" width="300">

where `A` is in red and `B` in blue.

## <a id="AddingRemoving"></a>Adding/removing/moving particles

Adding/removing/moving particles are order N operations.

These are shown in the `add_remove_move_particles.py` and `add_remove_move_particles_multispecies.py` scripts.
