# Arduino Q sensor reading.

the zip file is the Arduino lab app project the folder  
is the pyhton app client on host computer it leverage  
the python graphical user interface to display the sensor  
data of the dht11 temperature/moisture data via udp server.

identification between servers is done by sharing the same link.toml  
from there a shell command must be done on each side to config the  
server from the credential inside the link.toml file.  

inside the shared link.toml file  
   the ip is the one of the arduino board:  $ ip addr
the keygen(secret) is generated from 
   python3 ./udp_link/ keygen

copy and paste in the link.toml (it's important to respect the formatting)
then from this file copied on both side run 

client side (host pc):
    ./.venv/bin/python -m apps.pc_client --config link.toml
server side via ssh or serioal 
    python3 -m app.sensor_server --config link.toml --simulate

restart the arduino board and the client/server should sync

installation process are in details in read me file.

