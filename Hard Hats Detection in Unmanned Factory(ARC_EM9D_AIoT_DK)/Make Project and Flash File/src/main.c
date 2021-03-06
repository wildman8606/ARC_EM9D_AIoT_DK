/**************************************************************************************************
    (C) COPYRIGHT, Himax Technologies, Inc. ALL RIGHTS RESERVED
    ------------------------------------------------------------------------
    File        : main.c
    Project     : WEI
    DATE        : 2018/10/01
    AUTHOR      : 902452
    BRIFE       : main function
    HISTORY     : Initial version - 2018/10/01 created by Will
    			: V1.0			  - 2018/11/13 support CLI
**************************************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "embARC.h"
#include "embARC_debug.h"
#include "board_config.h"
#include "arc_timer.h"
#include "hx_drv_spi_s.h"
#include "spi_slave_protocol.h"
#include "hardware_config.h"

#include "tflitemicro_algo.h"
#include "model_settings.h"

#include "synopsys_sdk_camera_drv.h"

#include "hx_drv_iic_m.h"
#include "hx_drv_iomux.h"
#include "SC16IS750_Bluepacket.h"

extern uint32_t g_wdma2_baseaddr;

int8_t test_img [kNumRows * kNumCols] = {0};
uint8_t output_img [kNumRows * kNumCols] = {0};
uint8_t output_height = kNumRows;
uint8_t output_width = kNumCols;

DEV_UART * uart1_ptr;
char str_buf[100];

#define TEST_UPPER 25
uint8_t test_cnt = 0;
uint8_t test_correct = 0;
#include "test_samples.h"

int gpio_state;

int main(void)
{

    printf("\n\nversion 1\n\n");

    uart1_ptr = hx_drv_uart_get_dev(USE_SS_UART_1);
    uart1_ptr->uart_open(UART_BAUDRATE_115200);

    synopsys_camera_init();

    tflitemicro_algo_init();

    sprintf(str_buf, "Start While Loop\r\n");
    printf("into %s-%d\r\n", __func__, __LINE__);
    HX_GPIOSetup();
    IRQSetup();
    UartInit(SC16IS750_PROTOCOL_SPI);
    InitGPIOSetup(SC16IS750_PROTOCOL_SPI);
    uart1_ptr->uart_write(str_buf, strlen(str_buf));
    board_delay_ms(1000);
    while(test_cnt < TEST_UPPER)
    {
        gpio_state = GPIOGetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO7);
        if(gpio_state == 1){
		printf("Start_to_Capture\n");   
		sprintf(str_buf, "Start_to_Capture\r\n"); 
		uart1_ptr->uart_write(str_buf, strlen(str_buf));

		synopsys_camera_start_capture();
		board_delay_ms(100);

		uint8_t * img_ptr;
		uint32_t img_width = 640;
		uint32_t img_height = 480;
		img_ptr = (uint8_t *) g_wdma2_baseaddr;

		synopsys_camera_down_scaling(img_ptr, img_width, img_height, &output_img[0], output_width, output_height);

		for(uint32_t pixel_index = 0; pixel_index < kImageSize; pixel_index ++)
		    test_img[pixel_index] = output_img[pixel_index] - 128;

		int32_t test_result = tflitemicro_algo_run(&test_img[0]);
		printf("Answer: %2d\r\n", test_result);
		if(test_result == 1){
			printf("Wear hard hat, safe~\n");
			GPIOSetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO0, 0);
			GPIOSetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO1, 1);
		}else{
			printf("No Wear hard hat !!\n");
			GPIOSetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO0, 1);
			GPIOSetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO1, 0);
		}
		sprintf(str_buf, "Answer: %2d\r\n\n", test_result);
		uart1_ptr->uart_write(str_buf, strlen(str_buf));
		board_delay_ms(500);
		GPIOSetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO0, 0);
		GPIOSetPinState(SC16IS750_PROTOCOL_SPI, CH_A, GPIO1, 0);
	}else{
		printf("check\n");
	}
    }

	return 0;
}

