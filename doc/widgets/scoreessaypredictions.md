Score essay predictions
=======

Calculate exact agreement and quadratic weighted kappa.

**Inputs**
-  True scores and predictions: Datatable containing true and predicted scores. Most commonly output from 'Test and score' widget.

**Outputs**
-  Score: Calculated scores in datatable format.


**Score essay predictions** widget enables us to calculate domain specific scores for our models - exact agreement (percentage of predicted scores that match true scores exactly) and quadratic weighted kappa.

TODO slika <!-- ![](images/GeoMap-stamped.png) -->

1. Bring output data from 'Test and score' widget to input of Score essays.

2. The widget will try to detect true and predicted score attributes from input. Verify and change them using the dropdowns as necessary.

3. Calculated exact agreement and predicted scores will appear in the 'Results' section of the widget. They also get sent to output as a single datatable.

Examples
--------
TODO
<!--

In the first example we will model class predictions on a map. We will use *philadelphia-crime* data set, load it with **File** widget and connect it to **Map**. We can already observe the mapped points in Map. Now, we connect **Tree** to Map and set target variable to Type. This will display the predicted type of crime for a specific region of Philadelphia city (each region will be colored with a corresponding color code, explained in a legend on the right).

![](images/GeoMap-classification.png)

The second example uses [global-airports.csv](https://raw.githubusercontent.com/ajdapretnar/datasets/master/data/global_airports.csv) data. Say we somehow want to predict the altitude of the area based soley on the latitude and longitude. We again load the data with **File** widget and connect it to Map. Then we use a regressor, say, **kNN** and connect it to Map as well. Now we set target to altitude and use Black and White map type. The model guessed the Himalaya, but mades some errors elsewhere.

![](images/GeoMap-regression.png)

-->