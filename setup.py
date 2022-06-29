import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resolution_estimation_with_dl",
    version="0.0.1",
    author="Daniil Litvinov",
    author_email="danon6868@gmail.com",
    description="Pachage and scripts for local resolution estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danon6868/CryoEM_Resolution_Estimation",
    project_urls={
        "Bug Tracker": "https://github.com/danon6868/CryoEM_Resolution_Estimation/issues"
    },
    packages=["resolution_estimation_with_dl"],
)
