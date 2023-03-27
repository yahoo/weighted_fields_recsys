import setuptools

setuptools.setup(
     name='weighted_fields_recsys',
     version='0.1',
     author="Alex Shtoff",
     author_email="ashtoff@yahooinc.com",
     description="PyTorch classes for recommender systems with weighted multi-value fields",
     long_description="""
     PyTorch embedding bag module and factorization machine models for multi-value fields with weights per value. 
     For example, imagine a data-set of movies where the "genres" column may contain a list of genres with 
     corresponding weights representing a measure of confidence in the movie belonging to the genre.
     """,
     long_description_content_type="text/markdown",
     url="https://github.com/yahoo/weighted_fields_recsys",
     packages=['wfm'],
     install_requires=['torch>=1.13.1'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )