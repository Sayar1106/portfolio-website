---
title: 'Hosting FastAPI with Saturn Cloud Deployments'
subtitle: 'Seemlessly deploy your Machine Learning APIs'
summary: Seemlessly deploy your Machine Learning APIs
authors:
- admin
tags:
- cloud
categories: []
date: "2021-11-19T00:00:00Z"
lastmod: "2021-11-19T00:00:00Z"
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  caption: 'Image credit: [**Boitumelo Phetla**](https://unsplash.com/@writecodenow)'
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []

# Set captions for image gallery.
gallery_item:
- album: gallery
  caption: Default
  image: theme-default.png
- album: gallery
  caption: Ocean
  image: theme-ocean.png
- album: gallery
  caption: Forest
  image: theme-forest.png
- album: gallery
  caption: Dark
  image: theme-dark.png
- album: gallery
  caption: Apogee
  image: theme-apogee.png
- album: gallery
  caption: 1950s
  image: theme-1950s.png
- album: gallery
  caption: Coffee theme with Playfair font
  image: theme-coffee-playfair.png
- album: gallery
  caption: Strawberry
  image: theme-strawberry.png
---
---
Hi all, this article will explore the process of deploying a FastAPI application on Saturn Cloud. FastAPI is a robust web framework for building APIs with the Python language. [Saturn Cloud](https://saturncloud.io?utm_source=Sayar+Medium&utm_medium=FastAPI+Blog&utm_campaign=FastAPI+Blog) is a platform dedicated to scaling Machine Learning and Big Data pipelines and more.

The model will predict median house prices in California. Let's jump right into it.

### Resources

👉 [Repository](https://github.com/Sayar1106/cali-house-prices-estimator)

👉 [FastAPI](https://fastapi.tiangolo.com/)

👉 [Scikit-learn](https://scikit-learn.org/stable/)

👉 [Joblib](https://joblib.readthedocs.io/en/latest/)

---
### Data Exploration

The dataset I will use for training our machine learning model is called "California Housing Prices." It can be found [here](https://www.kaggle.com/camnugent/california-housing-prices).

The contents of our data are as follows:

The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. Be warned, the data isn't clean, so there are some preprocessing steps required! The columns are as follows, and their names are pretty self-explanatory:

*   longitude
*   latitude
*   housing\_median\_age
*   total\_rooms
*   total\_bedrooms
*   population
*   households
*   median\_income
*   median\_house\_ value
*   ocean\_proximity

On doing a rudimentary exploration of the dataset, I found the following:

![](/posts_img/fastapi_saturn/img_1.png)

A correlation plot between all the numerical features shows us the following:

![](/posts_img/fastapi_saturn/img_2.png)

According to the graph, most numerical features have very little correlation with median\_house\_value except median\_income, which seems to have a strong positive correlation of around 0.68.

---
### Data Cleaning/Feature Engineering

Since the total\_bedrooms feature had missing values, I had to impute it. For simplicity, I chose the median as the metric to impute the feature.

Additionally, two new features were engineered, namely, "rooms\_per\_households" and "population\_per\_household."

![](/posts_img/fastapi_saturn/img_3.png)

---
### Training the Model

Our repository looks like this:

![](/posts_img/fastapi_saturn/img_4.png)

The requirements.txt file contains our dependencies. It is crucial to have all the dependencies added to this file as it will be used during our deployment.

![](/posts_img/fastapi_saturn/img_5.png)

The file src/main.py contains our training script. Let us take a look at some of the essential functions in the script.

Our training model pipeline is relatively standard. There is just one categorical column (ocean\_proximity). For the other numerical columns, I applied a standard scaler. The ColumnTransformer estimator helps to facilitate feature transformations on heterogeneous data.

As for the model, I chose the Random Forest algorithm. I created the pipeline using scikit-learn's Pipeline class.

![](/posts_img/fastapi_saturn/img_6.png)

I used joblib to save our model. Since the model file was quite large (>100Mb), I decided to store it in AWS S3. The model's R² score was around 0.81, and the RMSE was around 49k.

---
### Setting up FastAPI Server and Frontend

As you may have guessed, app/main.py contains our code for the server. Since the model is stored in AWS, I used boto3 to download a local copy to the server.

If your bucket and file are private, you may need to set up authentication to access it on Saturn Cloud. You can do it by following this [guide](https://saturncloud.io/docs/using-saturn-cloud/connect_data/).

I wrote a simple function to load our model from AWS:

![](/posts_img/fastapi_saturn/img_7.png)

The variables BUCKET\_NAME and FILE\_NAME are self-explanatory. LOCAL\_PATH is the path to where the model will be copied locally.

I also defined global variables for the app, model, and templates.

![](/posts_img/fastapi_saturn/img_8.png)

#### Homepage

Since I'm creating an application, it's essential to have a homepage to serve as an interface for the model server.

I created a homepage for the app so that users can enter values for each of the features. To render the page, I used Jinja2Templates, which is provided out of the box by FastAPI templates.TemplateResponse renders our landing page titled "index.html."

![](/posts_img/fastapi_saturn/img_9.png)

index.html contains a form that will serve as the frontend for our application. The body of the page looks like this:

![](/posts_img/fastapi_saturn/img_10.png)

If you look closely at the form tag, you will see that the action attribute is set to "/submitform" and the request method is a POST request.

![](/posts_img/fastapi_saturn/img_11.png)

Our FastAPI server needs to have a method that handles the form data. This method needs to be decorated by app.post("/submitform") to handle the request appropriately.

![](/posts_img/fastapi_saturn/img_12.png)

You will notice that each of the variables is set as form parameters using Form. This class tells FastAPI that each variable's input is being received from a form.

You will also notice that line 26 has a method called predict. This method is actually where the model pipeline is fed the input from the form using the appropriate format. Since the pipeline can only receive input from a data frame, I first convert the data into a data frame. I then created the features as part of the feature engineering process. Finally, I return the model's predictions.

![](/posts_img/fastapi_saturn/img_13.png)

Once I had the price prediction, I used templates.TemplateResponse again to return a page called result.html. Along with "request", I also passed "price" through the TemplateResponse method. Finally, I rendered the price on the body of result.html.

![](/posts_img/fastapi_saturn/img_14.png)

---
### Deploying to Saturn Cloud

Before setting up the deployment, I pushed all of the code to Github. To deploy it, you must have your repository connected to Saturn Cloud. To do so, you can follow this [guide](https://saturncloud.io/docs/using-saturn-cloud/gitrepo/).

Once your repo is connected, head over to resources and select "New Deployment".

![](/posts_img/fastapi_saturn/img_15.png)

After this, you will be greeted with a form:

![](/posts_img/fastapi_saturn/img_16.png)

There are a few things to note when filling out the form. For instance, the "Command" is what the deployment will run to start your application.

![](/posts_img/fastapi_saturn/img_17.png)

Note that Saturn Cloud requires your applications to listen using port 8000.

Also, note the Extra Packages header. This is the script that will be used to install additional packages before the command is run. Since Saturn Cloud's default image does not have certain libraries like FastAPI and Uvicorn, pass "-r requirements.txt" to the text box.

This ensures that the script "\`pip install -r requirements.txt\` "is run before startup, containing dependencies for the additional packages.

Note that you can also write the individual names of each package in this section to install them.

Once you hit the Create button, your deployment will be created. Click on it and add your Github repo to the deployment. Ensure that you add the path to the Github resource to your working directory. Once that is done, click the green arrow to start the deployment.

![](/posts_img/fastapi_saturn/img_18.png)

Once your deployment is ready, click on the public URL. You should see a page like this:

![](/posts_img/fastapi_saturn/img_19.png)

Once you fill out the form, you will see a page with the predicted price:

![](/posts_img/fastapi_saturn/img_20.png)


Note that I used the last example of my test set as input. The actual median house price was $133000, so the model did a reasonably good job! 😀

**👉**  [**Link to the Github directory**](https://github.com/Sayar1106/cali-house-prices-estimator)

---
### Conclusion

Congratulations! You have successfully learned how to deploy a FastAPI model on [Saturn Cloud](https://saturncloud.io?utm_source=Sayar+Medium&utm_medium=FastAPI+Blog&utm_campaign=FastAPI+Blog)! If you're curious about using their environment, they offer 30 free hours a month for data scientists and teams. I hope you enjoyed reading this article. Until next time! ✋
