#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main( int argc, char * argv[] )
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help,h", "produce help message")
	;

	// Parse the command line options
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	// Handle help requests
	if (vm.count("help")) {
	    std::cerr << desc << "\n";
	    return 1;
	}
}
