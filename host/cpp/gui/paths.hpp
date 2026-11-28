// Where the app finds its files. A released app carries them next to its
// executable (demo/, fpga/program.tcl) and finds the bitstreams in a
// bitstreams/ folder beside it; a build from the source tree falls back to the
// repository (host/cpp/demo, fpga/, build/).
#pragma once
#include <QString>

// The models: demo/ next to the executable, else host/cpp/demo.
QString demoDir();

// fpga/program.tcl next to the executable, else the repository's.
QString programScript();

// The folder holding <array>/BoardTop.bit: bitstreams/ beside the app (beside
// the .AppImage file on Linux), else the repository's build/.
QString bitstreamDir();
