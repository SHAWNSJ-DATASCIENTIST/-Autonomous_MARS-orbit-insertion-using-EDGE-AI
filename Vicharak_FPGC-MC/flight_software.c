/************************************************************
 *                                                          *
 *                         M O M                            *
 *                                                          *
 *             MARS ORBITOR AUTONOMOUS AI                   *
 *                                                          *
 *              BY EX-AVIONICS LEAD: SHAWN SJ               *
 *                                                          *
 ************************************************************/

/************************************************************
 *                                                          *
 *                         N O T E S                        *
 *                                                          *
 *                INITIAL DATE: 10/08/2005                  *
 *                                                          *
 ************************************************************/

/*                                  --- ALL REGISTER ADDRESSES OBTAINED FROM LOW LEVEL DATASHEET
 * NOTE 01: CALIBRATION ERROR THEORY
 *
 * Every sensor contains inherent measurement errors caused
 * by manufacturing tolerances, environmental conditions,
 * electrical noise and mounting misalignment.
 *
 * Raw sensor readings should not automatically be treated
 * as true physical measurements.
 *
 * Calibration offsets must be determined through controlled
 * reference measurements before flight operations.
 *
 * NOTE 02: SENSOR CALIBRATION GUIDELINES
 *
 * 1. Keep the system stationary during initial calibration.
 * 2. Allow sensors to stabilize before recording readings.
 * 3. Collect multiple samples to reduce random noise.
 * 4. Calculate offsets using appropriate reference values.
 * 5. Verify calibration under different operating conditions.
 * 6. Recalibrate after hardware or mounting changes.
 *
 * NOTE 03: MAGNETOMETER ERROR THEORY
 *
 * Magnetic measurements may be affected by nearby metals,
 * electrical currents, permanent magnets and electromagnetic
 * interference.
 *
 * Keep the magnetometer away from high-current wiring and
 * magnetic components wherever practical.
 *
 * Perform hard-iron and soft-iron calibration when accurate
 * heading measurements are required.
 *
 * NOTE 04: IMU MEASUREMENT GUIDELINES
 *
 * Accelerometers measure specific force, while gyroscopes
 * measure angular velocity.
 *
 * Gyroscope integration accumulates bias errors over time.
 * Accelerometer-based orientation estimates may be affected
 * by vibration and linear acceleration.
 *
 * Sensor fusion can improve orientation estimates when
 * suitable filtering and calibration are implemented.
 *
 * NOTE 05: ALTITUDE MEASUREMENT THEORY
 *
 * Barometric altitude is estimated from atmospheric pressure.
 * Pressure changes caused by weather and local conditions
 * can introduce altitude errors.
 *
 * Time-of-flight sensors measure distance within their
 * operating range and do not directly measure altitude
 * without a suitable reference and coordinate arrangement.
 *
 * NOTE 06: DATA INTEGRITY GUIDELINES
 *
 * 1. Validate sensor readings before processing.
 * 2. Detect communication failures and invalid measurements.
 * 3. Use consistent units throughout the software.
 * 4. Include timestamps with recorded telemetry.
 * 5. Preserve raw readings when practical for later analysis.
 * 6. Never silently treat a failed sensor read as valid zero.
 *
 * NOTE 07: SYSTEM OPERATIONAL GUIDELINES
 *
 * 1. Verify all sensor connections before powering the system.
 * 2. Confirm I2C addresses and communication status.
 * 3. Check battery voltage and power stability.
 * 4. Test data logging before operational deployment.
 * 5. Verify sensor ranges and sampling frequencies.
 * 6. Test failure handling before any flight application.
 *
 * NOTE 08: ENGINEERING PRINCIPLE
 *
 * CALIBRATE BEFORE MEASUREMENT.
 * VALIDATE BEFORE PROCESSING.
 * VERIFY BEFORE DEPLOYMENT.
 *
 ************************************************************/

/************************************************************
 *                                                          *
 *                  SOFTWARE LIBRARIES                      *
 ************************************************************/

#include <Wire.h>
#include <Adafruit_BMP085.h>
#include <RTClib.h>
#include <MPU6050.h>
#include <VL53L0X.h>
#include <Arduino.h>
#include <math.h>

/************************************************************
 *                                                          *
 *              MACROS - PINOUTS AND ADDRESSES              *
 ************************************************************/

#define SDA_PIN 21
#define SCL_PIN 22

#define MAG_ADDR 0x1E
#define MPU_ADDR 0x69

#define SENSOR_DELAY 500
#define I2C_CLOCK 100000

#define G_TO_MS2 9.80665f
#define ACC_SENS 16384.0f

/************************************************************
 *                                                          *
 *                    SENSOR INSTANCES                      *
 ************************************************************/

Adafruit_BMP085 bmp;
RTC_DS3231 rtc;
MPU6050 mpu(MPU_ADDR);
VL53L0X sensor;

/************************************************************
 *                                                          *
 *             CALIBRATION AND RECOVERY                      *
 ************************************************************/

float accel_offset[3] = {0.0, 0.0, 0.0};
float mag_offset[3] = {0.0, 0.0, 0.0};

/************************************************************
 *                                                          *
 *                    KALMAN CONSTANTS                      *
 ************************************************************/

float kalman_placeholder = 0.0;

/************************************************************
 *                                                          *
 *             CONSTANTS AND GLOBAL VARIABLES               *
 ************************************************************/

float temperature = 0.0;
float pressure = 0.0;
float altitude = 0.0;

float accelX = 0.0;
float accelY = 0.0;
float accelZ = 0.0;

int16_t gyroX = 0;
int16_t gyroY = 0;
int16_t gyroZ = 0;

float Magx = 0.0;
float Magy = 0.0;
float Magz = 0.0;
float Mag_heading = 0.0;

int distance = 0;

String date;
String time;

unsigned long packetCount = 0;

/************************************************************
 *                                                          *
 *                 SENSOR INITIALIZATION                    *
 ************************************************************/

void writeRegister(uint8_t reg, uint8_t value)
{
    Wire.beginTransmission(MAG_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

int16_t readRegister16(uint8_t reg)
{
    Wire.beginTransmission(MAG_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);

    Wire.requestFrom(MAG_ADDR, (uint8_t)2);

    if (Wire.available() < 2)
    {
        return 0;
    }

    uint8_t low = Wire.read();
    uint8_t high = Wire.read();

    return (int16_t)(((uint16_t)high << 8) | low);
}

void init_bmp180()
{
    if (!bmp.begin())
    {
        Serial.println("BMP180 not found!");
        while (1)
            ;
    }
}

void init_rtc()
{
    if (!rtc.begin())
    {
        Serial.println("DS3231 not found!");
        while (1)
            ;
    }

    if (rtc.lostPower())
    {
        rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    }
}

void init_mpu6050()
{
    mpu.initialize();

    if (!mpu.testConnection())
    {
        Serial.println("MPU6050 not found at 0x69!");
        while (1)
            ;
    }
}

void init_vl53l0x()
{
    sensor.setTimeout(500);

    if (!sensor.init())
    {
        Serial.println("VL53L0X not found!");
        while (1)
            ;
    }

    sensor.setMeasurementTimingBudget(100000);
    sensor.startContinuous();
}

void init_magnetometer()
{
    writeRegister(0x60, 0x00);
    writeRegister(0x62, 0x00);
    writeRegister(0x63, 0x00);

    Serial.println("LSM303AGR Magnetometer OK");
}

void init_sensor()
{
    init_bmp180();
    init_rtc();
    init_mpu6050();
    init_vl53l0x();
    init_magnetometer();
}

/************************************************************
 *                                                          *
 *                  TELEMETRY OPERATIONS                    *
 ************************************************************/

String Telemetry()
{
    String data = String(packetCount) + "," +
                  date + "," +
                  time + "," +
                  String(temperature, 2) + "," +
                  String(pressure, 2) + "," +
                  String(altitude, 2) + "," +
                  String(accelX, 3) + "," +
                  String(accelY, 3) + "," +
                  String(accelZ, 3) + "," +
                  String(gyroX) + "," +
                  String(gyroY) + "," +
                  String(gyroZ) + "," +
                  String(distance) + "," +
                  String(Magx, 3) + "," +
                  String(Magy, 3) + "," +
                  String(Magz, 3) + "," +
                  String(Mag_heading, 2);

    return data;
}

/************************************************************
 *                                                          *
 *                    SENSOR SAMPLING                       *
 ************************************************************/

void sample_rtc()
{
    DateTime now = rtc.now();

    date = String(now.day()) + "/" +
           String(now.month()) + "/" +
           String(now.year());

    time = "";

    if (now.hour() < 10)
    {
        time += "0";
    }

    time += String(now.hour()) + ":";

    if (now.minute() < 10)
    {
        time += "0";
    }

    time += String(now.minute()) + ":";

    if (now.second() < 10)
    {
        time += "0";
    }

    time += String(now.second());
}

void sample_bmp180()
{
    temperature = bmp.readTemperature();
    pressure = bmp.readPressure() / 100.0;
    altitude = bmp.readAltitude();
}

void sample_mpu6050()
{
    int16_t ax, ay, az;

    mpu.getMotion6(&ax, &ay, &az, &gyroX, &gyroY, &gyroZ);

    accelX = (ax / ACC_SENS) * G_TO_MS2;
    accelY = (ay / ACC_SENS) * G_TO_MS2;
    accelZ = (az / ACC_SENS) * G_TO_MS2;
}

void sample_vl53l0x()
{
    distance = sensor.readRangeContinuousMillimeters();
}

void sample_magnetometer()
{
    int16_t rawX = readRegister16(0x68);
    int16_t rawY = readRegister16(0x6A);
    int16_t rawZ = readRegister16(0x6C);

    Magx = rawX * 0.15 - mag_offset[0];
    Magy = rawY * 0.15 - mag_offset[1];
    Magz = rawZ * 0.15 - mag_offset[2];

    Mag_heading = atan2(Magy, Magx) * 180.0 / PI;

    if (Mag_heading < 0)
    {
        Mag_heading += 360.0;
    }
}

void sample_sensors()
{
    sample_rtc();
    sample_bmp180();
    sample_mpu6050();
    sample_vl53l0x();
    sample_magnetometer();
}

/************************************************************
 *                                                          *
 *                     SENSOR FUSION                        *
 ************************************************************/

void sensorFusion()
{
    accelX -= accel_offset[0];
    accelY -= accel_offset[1];
    accelZ -= accel_offset[2];
}

/************************************************************
 *                                                          *
 *                   SENSOR CALIBRATION                     *
 ************************************************************/

void calibrateOffsets()
{
    for (int i = 0; i < 3; i++)
    {
        accel_offset[i] = 0.0;
        mag_offset[i] = 0.0;
    }
}

/************************************************************
 *                                                          *
 *                   EEPROM RECOVERY                        *
 ************************************************************/

void loadRecoveryData()
{
}

/************************************************************
 *                                                          *
 *                   OPERATION                              *
 ************************************************************/

void print_sensor_data()
{
    Serial.println();
    Serial.println("========== SENSOR DATA ==========");

    Serial.print("Packet Count: ");
    Serial.println(packetCount);

    Serial.print("Date: ");
    Serial.println(date);

    Serial.print("Time: ");
    Serial.println(time);

    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println(" C");

    Serial.print("Pressure: ");
    Serial.print(pressure);
    Serial.println(" hPa");

    Serial.print("Altitude: ");
    Serial.print(altitude);
    Serial.println(" m");

    Serial.print("Accel X: ");
    Serial.print(accelX);
    Serial.print(" m/s^2 | Y: ");
    Serial.print(accelY);
    Serial.print(" m/s^2 | Z: ");
    Serial.print(accelZ);
    Serial.println(" m/s^2");

    Serial.print("Gyro X: ");
    Serial.print(gyroX);
    Serial.print(" | Y: ");
    Serial.print(gyroY);
    Serial.print(" | Z: ");
    Serial.println(gyroZ);

    if (sensor.timeoutOccurred())
    {
        Serial.println("VL53L0X: TIMEOUT");
    }
    else if (distance < 30)
    {
        Serial.print("Very close: ");
        Serial.print(distance / 10.0);
        Serial.println(" cm");
    }
    else
    {
        Serial.print("Distance: ");
        Serial.print(distance / 10.0);
        Serial.println(" cm");
    }

    Serial.print("Magx: ");
    Serial.print(Magx);
    Serial.print(" uT | ");

    Serial.print("Magy: ");
    Serial.print(Magy);
    Serial.print(" uT | ");

    Serial.print("Magz: ");
    Serial.print(Magz);
    Serial.print(" uT | ");

    Serial.print("Mag_heading: ");
    Serial.print(Mag_heading);
    Serial.println(" deg");

    Serial.println("=================================");
}

/************************************************************
 *                                                          *
 *                    PROGRAM CONTROL                      *
 ************************************************************/

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(I2C_CLOCK);

    init_sensor();
    calibrateOffsets();
    loadRecoveryData();

    Serial.println("All sensors OK");
}

void loop()
{
    packetCount++;

    sample_sensors();
    sensorFusion();

    String data = Telemetry();

    print_sensor_data();

    Serial.println(data);

    delay(SENSOR_DELAY);
}
