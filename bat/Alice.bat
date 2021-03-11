set DATABASE_URL sqlite:///../apps/node/src/app/databasenode.db
conda activate syft-test & python ../apps/node/src/__main__.py -p 5000 --host 0.0.0.0 --start_local_db --id Alice