* VTC curve — clean inverter no access transistor
.include "/home/agarw/OpenRAM/technology/freepdk45/models/tran_models/models_nom/NMOS_VTG.inc"
.include "/home/agarw/OpenRAM/technology/freepdk45/models/tran_models/models_nom/PMOS_VTG.inc"

Mn1  Vout Vin  0   0   NMOS_VTG W=205.00n L=50n
Mp1  Vout Vin  vdd vdd PMOS_VTG W=90n    L=50n

Vvdd vdd 0 DC 1.0
Vin  Vin  0 DC 0

.dc Vin 0 1.0 0.001

.control
    run
    wrdata vtc1.txt v(Vin) v(Vout)
    quit
.endc
.end