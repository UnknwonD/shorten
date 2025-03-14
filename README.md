
# Shorten: 숏폼 동영상 편집 웹 서비스

**Shorten**은 Flask를 기반으로 간단히 구현된 웹 서비스입니다. 이 서비스는 프로토타입 단계로, 현재 숏폼 동영상 편집 기능만 제공합니다.

## 기능

- **숏폼 동영상 편집**: 업로드된 동영상을 자동으로 편집하여 숏폼 콘텐츠로 변환합니다. *(프로토타입 기능)*

## 실행 방법
* 학습된 모델 파일은 별도로 첨부되어 있지 않습니다.
* 관련 내용은 하단의 **개발코드 저장소**를 확인해주세요

1. **필수 패키지 설치**  
   프로젝트에 필요한 패키지를 아래 명령어로 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

2. **Flask 애플리케이션 실행**  
   아래 명령어를 실행하여 웹 서비스를 시작합니다.
   ```bash
   python app.py
   ```

3. **웹 브라우저에서 접속**  
   애플리케이션이 실행되면 웹 브라우저에서 아래 주소로 접속합니다.  
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## 파일 구조

```
.
├── app.py             # Flask 애플리케이션의 메인 파일
├── process.py         # 동영상 편집 기능 관련 모듈
├── requirements.txt   # 프로젝트 의존성 목록
├── templates/         # HTML 템플릿 파일들
└── static/            # 정적 파일 (CSS, JS, 이미지 등)
```

## 참고 사항

현재 이 서비스는 프로토타입으로, 숏폼 동영상 편집 기능만 제공됩니다. 추후 업데이트를 통해 정식 배포가 진행될 예정입니다.
- 개발 진행 코드는 아래 링크를 통해 확인해주세요
   - [개발 코드 라인 모음](https://github.com/UnknwonD/project_shorts)

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 확인해주세요.

