##
# Function which loads the packages passed as input
# Adapted from: https://stackoverflow.com/questions/38928326/is-there-something-like-requirements-txt-for-r
#
# Args:
# - packages [str]: packages list
#
installPackages <- function(packages = "default") {
	# install default packages if required
	if(length(packages) == 1L && packages == "default") {
		packages <- c(
			"plyr", 
			"dplyr", 
      "tibble",
			"devtools",
			"ggplot2", 
			"utils",
			"ggpmisc",
			"optparse"
		)
	}
	packagecheck <- match(packages, utils::installed.packages()[,1])
	# packages to install
	packagestoinstall <- packages[ is.na( packagecheck ) ]
	if(length( packagestoinstall ) > 0L) {
		print("Installing the requested packages...")
		utils::install.packages(
			packagestoinstall, 
			repos = "https://cran.mirror.garr.it/CRAN/"
		)
	}else{
		print("All requested packages are already installed")
	}
}

# Install packages
installPackages()