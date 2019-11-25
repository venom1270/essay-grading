Attribute selection
=======

Show data points on a world map.

**Inputs**
-  Graded essays: Corpus of graded essays ("train set").
-  Ungraded essays: Corpus of ungraded essays ("test set").
-  Source texts: Optional corpus of source texts (text essays are based on, i.e. a story or instructions).

**Outputs**
-  Graded attributes: Attributes for graded essay corpus.
-  Ungraded attributes: Attributes for ungraded essay corpus.


**Attribute selection** widget gives us the option to select and calculate desired attributes from input essays. All inputs are of type 'Corpus' (from Orange-text), while outputs are of type 'DataTable'. We can then use these outputs in a standard Orange fashion (models, predictions ...).

TODO slika <!-- ![](images/GeoMap-stamped.png) -->

1. Select desired attributes

   - Select desired attributes using the available checkboxes
   - There are 6 categories of attributes which include over 70 attributes in total
        - Basic measures:
            - Number of characters
            - Number of words
            - Number of short words
            - Number of long words
            - Most frequent word length
            - Average word length
            - Number of sentences
            - Number of short sentences
            - Number of long sentences
            - Most frequent sentence length
            - Average sentence length
            - Number of different words
            - Number of stopwords
        - Readability measures:
            - Gunning Fog index
            - Flesch reading ease index
            - Flesch Kincaid grade
            - Dale Chall readability formula
            - Automated readability index
            - Simple measure of Gobbledygook
            - LIX
            - Word variation index
            - Nominal ratio
        - Lexical diversity:
            - Type-token ration
            - Guiraud's index
            - Yule's K
            - The D estimate
            - Hapax legomena
            - Advanced Guiraud's index
        - Grammar:
            - Number of each different POS tag (~30 attributes)
            - Average sentence structure tree height
            - Verb form TODO?
        - Content:
            - Number of spellchecking errors
            - Number of capitalization errors
            - Number of punctuation errors
            - Cosine similarity with source text (if source text present)
            - Grade that the current essay's cosine similarity is most similar to
            - Cosine similarity with best essays
            - Cosine pattern
            - Cosine correlation values
        - Coherence:
            - Avg/min/max distance to neighbouring points (2x, euc. and cos. distance)
            - Avg/min/max distance to any point (2x, euc. and cos. distance)
            - Clark Evans nearest neighbour
            - Average distance of nearest neighbour
            - Frequency TODO
            - Avg/min/max distance to centroid (2x, euc. and cos. distance)
            - Standard distance
            - Relative distance
            - Determinant of distance matrix
            - Moran's I
            - Geary's C
            - Gettis' G

2. Select word embeddings:

   - Word embeddings are used during calculations of 'Coherence' attributes
   - Choose 'TF-IDF' or SpaCy's 'GloVe' word embeddings

3. The calculation may take a few minutes, depending on attribute categories selected. 'Grammar', 'Content' and 'Coherence' are most demanding.

4. Due to speed, selection changes are NOT communicated automatically. You can change this by ticking the checkbox next to 'Apply' button.

Examples
--------
TODO
<!--

In the first example we will model class predictions on a map. We will use *philadelphia-crime* data set, load it with **File** widget and connect it to **Map**. We can already observe the mapped points in Map. Now, we connect **Tree** to Map and set target variable to Type. This will display the predicted type of crime for a specific region of Philadelphia city (each region will be colored with a corresponding color code, explained in a legend on the right).

![](images/GeoMap-classification.png)

The second example uses [global-airports.csv](https://raw.githubusercontent.com/ajdapretnar/datasets/master/data/global_airports.csv) data. Say we somehow want to predict the altitude of the area based soley on the latitude and longitude. We again load the data with **File** widget and connect it to Map. Then we use a regressor, say, **kNN** and connect it to Map as well. Now we set target to altitude and use Black and White map type. The model guessed the Himalaya, but mades some errors elsewhere.

![](images/GeoMap-regression.png)

-->