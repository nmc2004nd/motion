#include <AccelStepper.h>

#define STEP_PIN 3
#define DIR_PIN  4

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

const float MICRO_STEP    = 16.0;
const float ANGLE_STEP    = 1.8;
const float MM_PER_REV    = 8.0;
const float STEPS_PER_MM  = MICRO_STEP * (360.0 / ANGLE_STEP) / MM_PER_REV;

const byte BUFFER_SIZE = 32;
char rxBuf[BUFFER_SIZE];
bool newData = false;

void setup()
{
  Serial.begin(115200);
  stepper.setMaxSpeed(3200);
  stepper.setAcceleration(1000);
}

void loop()
{
  stepper.run();
  recvLine();
  if (newData) {
    parseCmd();
    newData = false;
  }
}

void recvLine()
{
  static byte idx = 0;
  while (Serial.available() > 0 && !newData) {
    char c = Serial.read();
    if (c == '\n') {
      rxBuf[idx] = '\0';
      idx = 0;
      newData = true;
    } else if (idx < BUFFER_SIZE - 1) {
      rxBuf[idx++] = c;
    }
  }
}

void parseCmd()
{
  char cmd = rxBuf[0];

  if (cmd == 's') {
    stepper.stop();
    Serial.println("M: Stopped");
    return;
  }

  if ((cmd == 'f' || cmd == 'b') && rxBuf[1] == ' ') {
    char *endPtr;
    float distance = strtof(&rxBuf[2], &endPtr);

    // strtof trả về con trỏ bằng start nếu không parse được
    if (endPtr == &rxBuf[2] || distance <= 0.0f) {
      Serial.println("ERR: invalid distance");
      return;
    }

    long steps = (long)round(distance * STEPS_PER_MM);
    if (cmd == 'f') {
      stepper.move(steps);
      Serial.print("M: Forward ");
    } else {
      stepper.move(-steps);
      Serial.print("M: Backward ");
    }
    Serial.print(distance);
    Serial.println(" mm");
    return;
  }

  Serial.println("ERR: unknown cmd");
}
