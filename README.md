# Measuring scanned leaf images using OpenCV

As part of a project for the University of Calgary's STAT 517 course, students were expected to measure leaf dimensions by hand.
This OpenCV script automates the process of measuring leaf dimensions and *was not a required part of the course.*


## Examples

Here is an example image, where the script has detected the following:
* longest and shortest width along all directions (blue)
* convex hull around each object and angles between each line segment (green and orange)
* length of perimeter and sum of total angles around convex hull (green)
* objects outside of a predefined size range are rejected (red)

![Example output from winnower-size.py](/out/cherry/cherry1-labelled.png)

The course project also required the use of linear discriminant analysis (LDA) to classify leaves by species.
Originally, students were expected to use the length and width of the leaves as features for LDA.
However, it turns out that the two sharpest angles in the convex hull (*which were not in the course project*) are much better predictors.

Here is a comparison of the angle features versus length/width.
Clearly, the two species are much easier to separate when plotted by their two sharpest angles. This is because the pear leaves have a rounded base, whereas the cherry leaves are pointed on both ends.

![Linear discriminant analysis using leaf angles](/cls/a_c-plot.png)
![Linear discriminant analysis using length and width](/cls/wl_c-plot.png)

## Running the scripts

`winnower-size.py` requires Python 3.8 and OpenCV, among other libraries. To install the required python packages:

```
pip install --upgrade pip
pip install -r requirements.txt
```

To run `winnower-size.py` on the examples in the directory `in`, and save images and measurements to `out`:
```
python winnower-size.py --input in --output out --scale 0.12 --display True
```

`winnower-lda.py` generates the plots and predictions using linear discriminant analysis.
To run `winnower-lda.py` on the output of `winnower-size.py`:
```
python winnower-lda.py --train out/measurements.csv --test out/measurements.csv --output results.csv --covariance True --display True
```

See the report `winnower-cv-report.pdf` for a full description of both scripts and their parameters.
