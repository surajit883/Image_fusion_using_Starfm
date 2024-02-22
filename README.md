Overview
The script utilizes the STARFM algorithm to fuse Landsat and MODIS NDVI data, providing a high temporal resolution composite.

Usage
To use the script, follow these steps:

Ensure you have Python 3 installed on your system.
Install the required dependencies listed in the "Dependencies" section.
Update the script with your desired start date, end date, and coordinates defining the area of interest.
Run the script using the command python script_name.py.
Parameters
start_date: Start date for data collection in 'YYYY-MM-DD' format.
end_date: End date for data collection in 'YYYY-MM-DD' format.
coord: List of coordinates defining the area of interest.
Dependencies
Earth Engine Python API: Python library for interacting with Google Earth Engine.
NumPy: Library for numerical computing in Python.
SciPy: Library for scientific computing and interpolation.
STARFM4Py: Python implementation of the STARFM algorithm.
Author
Surajit Hazra - Initial work
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This script was inspired by the STARFM algorithm and utilizes data from Google Earth Engine.
