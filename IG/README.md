# Codes to calculate IGs for cis−trans isomerization of azobenzen

## Functions to calculate IG

`interpolate_inputs`:
generate linear interpolation between the baseline and the original input

`compute_gradients`: 
compute gradients for each input to the model

`integral_appeoximation`: 
accumulate gradients along the alpha intervals

## Generate input
1. Generate `baseline` using the equilibrium geometry of azobenzene and setting velocity to 0
2. Generate original `inputs` of GICnet model using geometry near transition state and ts=10

## Load model
1. Load one GICnet model `fourd_tc10`

## Calculate IG
1. Set `m_steps` and generate values of `alphas` 
2. Calculate `IG`

## Code usage
1. Set path to GICnet model in `load model` part
2. Install MLatom and set the correct path in `sys.path`, so that you can use functions in MLatom to generate inputs for GICnet model
3. Activate proper Python environment for MLatom
4. Run `python IG/IG.py`


## Reference:
1. https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
2. Sundararajan, M.;  Taly, A.; Yan, Q. In Axiomatic attribution for deep networks, 34th International Conference on Machine Learning, ICML 2017, 2017; pp 5109–5118.
