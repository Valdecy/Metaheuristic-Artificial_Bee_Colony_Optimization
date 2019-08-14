# Metaheuristic-Artificial_Bee_Colony_Optimization
Artificial Bee Colony Optimization to Minimize Functions with Continuous Variables. The function returns: 1) An array containing the used value(s) for the target function and the output of the target function f(x). For example, if the function f(x1, x2) is used, then the array would be [x1, x2, f(x1, x2)].  


* food_sources = The population size. The Default Value is 3.

* min_values = The minimum value that the variable(s) from a list can have. The default value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The default value is  5.

* iterations = The total number of iterations. The Default Value is 50.

* employed_bees = Number of Employed Bees. The Default Value is 3.

* outlookers_bees = Number of Outlookers Bees. The Default Value is 3.

* limit = Scouter Bee improvement in food sources that stucked in local optima for more iterations than the limit value. The Default Value is 3.

* target_function = Function to be minimized.
