import cv2
import mediapipe as mp
import random as r
import time

# Initialize game options
game = ["rock", "paper", "scissors"]

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_obj = mp_hands.Hands(max_num_hands=1)

# Initialize variables
start_time = False
start_init = False 
prev = None  
w=None
user=None
computer=None


def choose(hand_keyPoints):
    cnt, cnt1 = 0, 0
    thresh = (hand_keyPoints.landmark[0].y * 100 - hand_keyPoints.landmark[9].y * 100) / 2

    # Check finger positions
    if (hand_keyPoints.landmark[5].y * 100 - hand_keyPoints.landmark[8].y * 100) > thresh:
        cnt += 1  
    if (hand_keyPoints.landmark[9].y * 100 - hand_keyPoints.landmark[12].y * 100) > thresh:
        cnt += 1  
    if (hand_keyPoints.landmark[13].y * 100 - hand_keyPoints.landmark[16].y * 100) > thresh:
        cnt1 += 1 
    if (hand_keyPoints.landmark[17].y * 100 - hand_keyPoints.landmark[20].y * 100) > thresh:
        cnt1 += 1  
    if (hand_keyPoints.landmark[5].x * 100 - hand_keyPoints.landmark[4].x * 100) > 6:
        cnt1 += 1  

    # Determine gesture
    if cnt + cnt1 == 5:
        return "paper"
    elif cnt == 2:
        return "scissors"
    elif cnt + cnt1 == 0:
        return "rock"
    else:
        return "error"


def winner(user, computer):
    if user == computer:
        return "Draw"
    elif (user == "rock" and computer == "scissors") or \
         (user == "scissors" and computer == "paper") or \
         (user == "paper" and computer == "rock"):
        return "User wins"
    else:
        return "Computer wins"


def main():
    global start_init, prev,w,user,computer  
    cap = cv2.VideoCapture(0)


    while True:
        
        end_int = time.time()
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # Display game title
        text = "ROCK-PAPER-SCISSORS"
        position = (125, 30)  
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)
        thickness = 5
        cv2.putText(frm, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        if res.multi_hand_landmarks:
            u = choose(res.multi_hand_landmarks[0])
            if u != prev: 
                if not start_init:
                    start_time = time.time()
                    start_init = True
                elif (end_int - start_time) > 0.2:
                    user = u
                    prev = u  
                    start_init = False

                    if user == "error":
                        cv2.putText(frm, "Make correct gesture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        computer = r.choice(game)
                        w = winner(user, computer)
                        
                        
        cv2.putText(frm, f"Your Gesture: {user}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frm, f"Computer's Gesture: {computer}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frm, w, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frm)
        prev=None
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()