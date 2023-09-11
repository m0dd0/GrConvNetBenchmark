import setuptools

setuptools.setup(
    name="grconvnet",
    version="0.0.1",
    author="Moritz Hesche",
    author_email="mo.hesche@gmail.com",
    # description="",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    classifiers=["Programming Language :: Python :: 3"],
    packages=setuptools.find_packages(),
    # some packages are not available for pytohn 3.11
    # some packages need typing modules which are only available for python >3.8
    # created using python==3.9
    python_requires=">=3.8, <3.11",
    # not specifying versions might result in pip downloading multiple versions
    # of a package in order to solve dependencies
    # therfore it might be useful to fix the versions someday
    install_requires=[
        "numpy==1.23.5", # 1.23.5 is the latest versions that is supported by ros_numpy
        "opencv-python",
        "matplotlib",
        "scikit-image",
        "imageio",
        "torch",
        "torchvision",
        "torchsummary",
        "tensorboardX",
        "Pillow",
        "open3d",
        "torchtyping",
        "nptyping",
        "tqdm",
    ],
    extras_require={"dev": ["black", "pylint", "jupyter", "ipykernel"]},
    include_package_data=True,
    use_scm_version=True,
)
