# integrated-dynamics

This is a work-in-progress dynamics library designed to provide accurate modelling of any motorized robot mechanism. The library uses ordinary differential equations to ensure correct integration even when the mechanism states change very rapidly between physics updates. For example, this library can show how rapidly a 30 pound load, connected to a 20:1 gearbox and a single cim motor, accelerates to max speed (3.47 feet/sec) from standstill (in less than a tenth of a second).

One use for this is toward a more-accurate off-robot simulation experience. If you want a metric ton of numbers, you can use this right now! For the lesser nerds among us, pyfrc compatibility is in the works.

Another big use of this is to provide on-robot automatic sensor fusion via an extended kalman filter. This combines simulation estimates with indirect, potentially noisy sensor data to approximate robot state in real time.

Something else in the repository is my python implementation of iLQG. In a nutshell, iLQG uses two functions -- a dynamics function and a cost function -- to generate a locally-optimal discrete-time state-space controller. In other words, tell it how your robot works, and what is beneficial for it to do, and it will spit out a list of gains that result in an awesome autonomous mode.