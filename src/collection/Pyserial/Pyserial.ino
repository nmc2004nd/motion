#include <AccelStepper.h>

#define STEP_PIN 3
#define DIR_PIN 4

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// Cấu hình quy đổi ra mm
float microStep = 16.0;                                                      // Vi bước (vd: 16)
float angleStep = 1.8;                                                       // Góc bước motor (thường 1.8 độ)
float distancePerRound = 8.0;                                                // Hành trình 1 vòng (bước vitme, vd 8mm)
float stepsPerDistance = microStep * (360.0 / angleStep) / distancePerRound; // VD: 3200/8 = 400

const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

void setup()
{
  Serial.begin(115200);

  stepper.setMaxSpeed(3200);     // tốc độ tối đa
  stepper.setAcceleration(1000); // gia tốc (càng cao càng "bốc")
}

void loop()
{
  // LUÔN PHẢI GỌI HÀM NÀY ĐỂ MOTOR CHẠY (Non-blocking)
  stepper.run();

  // Đọc dữ liệu từ Python
  recvWithEndMarker();
  // Xử lý lệnh nếu có dữ liệu mới
  parseData();
}

void recvWithEndMarker()
{
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;

  // Chỉ đọc khi có dữ liệu và chưa có dữ liệu mới chưa xử lý
  while (Serial.available() > 0 && newData == false)
  {
    rc = Serial.read();

    if (rc != endMarker)
    {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars)
      {
        ndx = numChars - 1;
      }
    }
    else
    {
      receivedChars[ndx] = '\0'; // Kết thúc chuỗi
      ndx = 0;
      newData = true;
    }
  }
}

void parseData()
{
  if (newData == true)
  {
    char cmd = receivedChars[0];
    float distance = 0.0;
    long steps = 0;

    // Ví dụ lệnh: "f 150" (150 mm)
    if (strlen(receivedChars) > 1)
    {
      distance = atof(&receivedChars[2]);         // Lấy giá trị khoảng cách (float)
      steps = round(distance * stepsPerDistance); // Tính số bước dựa trên công thức
    }

    if (cmd == 'f')
    {
      stepper.move(steps);
      Serial.print("M: Forward ");
      Serial.print(distance);
      Serial.println(" mm");
    }
    else if (cmd == 'b')
    {
      stepper.move(-steps);
      Serial.print("M: Backward ");
      Serial.print(distance);
      Serial.println(" mm");
    }
    else if (cmd == 's')
    {
      stepper.stop(); // Dừng mềm
      Serial.println("M: Stopped!");
    }

    // Đánh dấu đã xử lý xong
    newData = false;
  }
}