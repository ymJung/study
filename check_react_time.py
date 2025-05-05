import pygame
import random
import time
import sys

# 설정
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)  # 흰색
COLORS = {
    '빨강': (255, 0, 0),
    '초록': (0, 255, 0),
    '파랑': (0, 0, 255),
}
TRIALS = 5

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("반응 속도 측정")
    font = pygame.font.SysFont(None, 72)

    times = []

    for i in range(TRIALS):
        screen.fill(BACKGROUND_COLOR)
        pygame.display.flip()

        wait_ms = random.randint(1000, 3000)
        pygame.time.delay(wait_ms)   # ms 단위 대기

        # 색상 변경
        color_name, color = random.choice(list(COLORS.items()))
        start = time.perf_counter()
        screen.fill(color)
        pygame.display.flip()

        # 클릭 대기
        clicked = False
        while not clicked:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    end = time.perf_counter()
                    times.append(end - start)
                    clicked = True

        print(f"{i+1}번째 반응 시간 ({color_name}): {(times[-1]*1000):.1f} ms")

    # 평균 계산
    avg = sum(times) / len(times)

    # 평균 시간 화면 표시
    screen.fill(BACKGROUND_COLOR)
    text = font.render(f"평균 반응 시간: {avg*1000:.1f} ms", True, (0, 0, 0))
    rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
    screen.blit(text, rect)
    pygame.display.flip()

    # 종료 대기
    while True:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.MOUSEBUTTONDOWN):
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()
