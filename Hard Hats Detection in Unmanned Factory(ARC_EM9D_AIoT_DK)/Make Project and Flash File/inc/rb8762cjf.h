/*
	RB8762CJF programming define
*/

//Bluetooth AT Command List
#define	AT_CMD	"at\r\n"
#define	AT_CMD_READ_BT_ADDR	"at+laddr\r\n"

#define	AT_CMD_READ_BT_NAME	"at+name\r\n"
#define AT_CMD_WRITE_BT_NAME	"at+name=%s\r\n"

#define	AT_CMD_READ_BT_BAUDRATE	"at+baud\r\n"
#define AT_CMD_WRITE_BT_BAUDRATE	"at+baud=%d\r\n"

//role parameters: 0 for slave; 1 for master
#define AT_CMD_READ_BT_ROLE	"at+role\r\n"
#define AT_CMD_WRITE_BT_ROLE	"at+role=%d\r\n"

//Reboot device
#define AT_CMD_REBOOT_DEV	"at+reset\r\n"
//Reset the device to factory default.
#define AT_CMD_RETURN_TO_DEFAULT	"at+default\r\n"

#define AT_CMD_READ_VERSION	"at+version\r\n"

//0 for Manual broadcast, 1 for Automatic broadcast
#define AT_CMD_WRITE_ADVMODE	"at+advmode=%d\r\n"
#define AT_CMD_READ_ADVMODE	"at+advmode\r\n"

//0 for turn on broadcast, 1 for turn off broadcast
#define AT_CMD_WRITE_ADVEN	"at+adven=%d\r\n"

//broadcast cycle. Maximum 10240 ms, default setting is 200ms (hex 0140)
#define AT_CMD_WRITE_ADVINT	"at+advint=%s\r\n"
#define AT_CMD_READ_ADVINT	"at+advint\r\n"

/*-------------------------
parameter (-20, 0, 3, 4, 8)
-20 : -20dBm
0: 0dBm
3: 3dBm
4: 4dBm
8: 8dBm
--------------------------*/
#define AT_CMD_READ_POWER	"at+power\r\n"
#define AT_CMD_READ_POWER	"at+power=%s\r\n"


//only for Master mode
#define AT_CMD_MASTER_START_SCAN_DEVICE "at+inq\r\n"
#define AT_CMD_MASTER_STOP_SCAN_DEVICE	"at+sinq\r\n"
//Input parameter from "at+inq"
#define AT_CMD_MASTER_CON_TO_DEVICE	"at+conn=%d\r\n"

typedef enum BT_POWER_PARA
{
	BT_POWER_SET_M20_DB = -20,
	BT_POWER_SET_0_DB = 0,
	BT_POWER_SET_3_DB = 3,
	BT_POWER_SET_4_DB = 4,
	BT_POWER_SET_8_DB = 8

}BT_POWER;

typedef enum BT_ADV_MODE_PARA
{
	BT_ADV_MANUAL_MODE_0,
	BT_ADV_AUTOMATIC_MODE_1

}BT_ADV_MODE;


typedef enum BT_ADV_ON_OFF_PARA
{
	BT_ADV_TURN_OFF,
	BT_ADV_TURN_ON

}BT_ADV_ON_OFF;

typedef enum BT_BAUDRATE_PARA
{
	BT_BAUDRATE_9600=9600,
	BT_BAUDRATE_115200=115200,

}BT_BAUDRATE;

typedef enum BT_ROLE_PARA
{
	BT_ROLE_SLAVE,
	BT_ROLE_MASTER
}BT_ROLE;
