# CI/CD Notes

Personal notes and interpretation of this repo's CI/CD setup.

The actual workflow lives at the repo root, not in this project folder:
`data-science-portfolio/.github/workflows/ci.yml` — it's shared across
every project in the monorepo (`attribution/`, `customer-intelligence-platform/`,
`experimentation/`, `marketing-analytics/`, etc.), not scoped to this one.

MLOPs checklist for model architecture
Move from monolithic notebooks to modular source directory designs. Notebooks are for exploration, not production
Refactor the code into object oriented python
Putting this in standard src folder, then separate config files using yaml formats that dictate hyperparameter limits and data paths from logic. 
feature engg, model training and model evaluation metrics all live within testable modules within in the src folder. 

Unit testing--You are not testing if the model works, you are running property based tests on the pipeline itself

You write test to ensure catgeorical encoder isnt dropping unseen variable from the live datastream. 

You write schema test to ensure that upstream data team didnt suddencly change the column name from customer _order_date to cust_ord_dt

ML flow compatible artifcats and full login
Traceability to track exact model version back to specifc gut committ of the code to specific yaml config & the exact snapshot of the data it was trained on
ML flow provides that registry. All of this is orchestrated via CI/CD pipelines-continuous integration and continuous deployment. Its the engine of modern ML Ops. 

If DS pushed a new feature engg script to github, the CI/CD pipeline automatically spins up a container, runs the unit test, checks the schema validations and verifies the ml flow intgeration before it ever touches the live production environment. It creates an automated safety net, allowing the team to iterate rapidly without fear of breaking the production. 

Spark is a distributed architecture and relies on cluster node processign data in paralle. 





