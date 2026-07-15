from setuptools import find_packages, setup


setup(
    name="sms-lira-mtgf",
    version="0.1.0",
    description="Membership inference attack research framework for diffusion models using MTGF.",
    packages=find_packages(),
    py_modules=["config", "run_pipeline"],
    python_requires=">=3.10",
)
