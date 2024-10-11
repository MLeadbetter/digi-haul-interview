# Assumptions

For an actual project I would have talked to people or done further research to remove all of these.

 - Data should not be commited to public repo for potential confidentiality reasons.
 - GPS logging device stays with the driver, not with the truck. The driver may walk around with the device during a delivery.
 - Data is complete and not a sample
 - As the columns don't exist (but are clearly named to leave space for the missing ones) we only have point to point deliveries with all back hauls excluded.
 - Truck arrival time is considered "delivery time", rather than when the truck had been unloaded and is leaving.
 - Trucks that are "close" to the delivery site are delivering (1km was chosen to be "close").
 - The speed limit is 100kmph
 - Trucks take at least 15 minutes to deliver their cargo
 - As PROJECT_ID has not been described in the accompanying document, I assumed it shouldn't be in the data and its use has been ommited.
 - For the sake of brevity, I've pretended that concept drift doesn't exist and have validated on a random samples of the data. Though I'm happy to discuss how I'd fix this.
 - Trucks may be driven for up to 8 hours a day

# Additional work

It's possible to spend a very significant amount more time on this project. Here are some things that should be explored further for this project:

 - As I used the project in part to learn Polars, there's certainly parts of the code that I think can be improved since finishing.
 - Choosing constants based on analysis, rather than what I decided is probably fine
 - Add documentation to functions describing what they do.
 - Add unit tests to ensure everything works correctly and stays that way
 - Further data cleaning. I'm certain I didn't get every issue.
 - Visualising GPS points on a map and manually inspecting routes that seem odd. Adapting the delivery time detection to be more accurate from this. The implementation I made was very quick and dirty.
 - Accounting for concept drift. For best results this would go hand in hand with seasonality and would require more data. Especially going over December.
 - Hyperparameter tuning of the model
 - As we are supposed to predict the likelihood, checking probability curves and calibrating on additional hold out data may be necessary (though we may not have sufficient data if it's a bit out).

Additional data is also likely to be available (for example the number of satalites used in each GPS point) which can be used to increase the accuracy of the delivery time.

Adding in information about the road network (specifically traffic and distance measurements) are sure to improve the accuracy of the project.

# Deployment plan

There are different ways of deploying a Data Science model. They very much depend on what it will be used for, which is context that is missing from the assignment. It could be as a query-able micro service. It could be coded into a larger program that doesn't want to take a high performance hit from a network request. But probably the most common way is creating a docker container which is scheduled to run (via AirFlow or similar) that takes the data it needs from the relavent databases retrains as neccesary and sends a batch of predictions to another database to be consumed by other programs.

Regardless of choice, monitoring should also be set up to ensure that the inputs and outputs look reasonable, according to some simple rules
