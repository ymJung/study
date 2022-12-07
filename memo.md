
### 객체지향 
> 응집도를 높이고 결합도를 낮춘다.
### SOLID
    - SRP 1클1일 / OCP 인터페이스사용 / LSP 상위하위변환    / ISP / DIP
    >> SOC 관심사를 분리하면 SOLID는 자연스럽게 구현된다. / 파일갯수는 늘어나지만 유지보수는 쉬워진다.

### ocp? 개방 폐쇄의 원칙
- 변경될것과 변하지 않을것을 구분 
- 만나는지점 인터페이스 정의
- 정의한 인터페이스에 의존하게 코드를 작성
- 기능을 바꾸어도 코드는 그대로 - 왜?-인터페이스니깐

https://dublin-java.tistory.com/48


### proxy-aop
- 대행자
- 객체에 대한 접근을 제어하거나 [기능] 을 추가
    - ex. 변경하지 않고 시간을 체크 
- 상속extends  , interface 
- lazy initialize 적용 , 권한 체크 등 장점



### 부하 발생시 대처 순서
1. rdb라면 read  replication 로 read  write api를 분산해본다.
2. redis로 잦은 select할 오브젝트 캐시로 db부하를 완화시킨다
3. db를 대규모 트래픽에 대비해 스케일아웃이 가능하도록 nosql로 재 설계해보고
4. db가 뻗을거 대비해 빠른 복구가 가능한 db  proxy를 이용하거나 철저한 백업과 cdc기능으로 sync할 db를 준비해둔다.

### 캐시
- appcache - ESO캐시 웹캐시  redis cdn

## MSA
- 어떤 장점이 있는지
    - 빌드/배포 가 빠르고 대응이 빠르다.  - 장애 발생 시 복구가 쉽다.
    - 동일한 동작을 하는 어플리케이션을 재사용할수 있다. 
    - 책임이 명확한 컴포넌트로 나누어 도메인 전문성을 갖추게 된다. 응집도가 높아진다
    - 프로젝트 복잡도가 감소 한다. 결합도가 낮아진다. 
    - 로드밸런서의 기능을 수행할 수있다.
- 단점이 뭔지 
    - 트렌젝션 관리가 어렵다.  
    - 장애 발생시 추적이 어렵다. 
    - 관리자가 적을경우 모르는 서비스가 생긴다.  - 소스관리

* SAGA
    - 분산 트렌젝션 환경에서 영속성을 유지하기 위한 방법
    - 보상 트렌젝션을 실행 
    - 각각의 상태를 메시지 형태로
        * 코레오그래피 (choregraphy:안무)  : 의사결정을 참여자. 이벤트 교환 방식 통신
            - 중앙편성자가 없음. 다음참여자를 트리거 하는 이벤트 발행
            - publish subscribenm 형태로 소통
            * 주의점
                - 원자적으로 일어남
                - 트렌젝셔널 메시징 - 수신받은 데이터와 자신이 가진 데이터를 연관 지을수 있어야함.
            * 장점 : 비지니스별 객체를 생성 단순화, 느슨한 결합 - 이벤트를 구독할뿐 서로 모름
            * 단점 : 구현로직이 흩어져 이해하기 어려움, 모니터링이 어려움, 타임아웃이 어려움 
        * 오케스트레이션 (orchestration:관현악) : 의사결정을 중앙화. 참여자에게 커맨드 메시지를 보냄
            - 커맨드 비동기 응답 상호방식으올 동작
            - 메시지 브로커가 전송
            * 장점: 의존관계 단순화 / 비지니스로직을 단순화
            * 단점 : 잘못된 중앙화 유발 , 중앙서비스가 죽으면 서비스 이용불가
            - 이미 구현된 서비스에 적용하기 좋음 
* CQRS
    > Command and Query Responsibility Segregation(명령과 조회의 책임 분리)

    - CQRS는 시스템의 상태를 변경하는 작업과 시스템의 상태를 반환하는 작업의 책임을 분리하는 것입니다.
    - CQRS는 데이터를 업데이트하는 명령과 데이터를 읽는 쿼리 를 사용하여 읽기 및 쓰기를 다른 모델로 구분 합니다 .
    - 시스템의 제한된 구역에 CQRS 적용을 고려
    - 시간이 흐름에 따라 하나의 모델이 점점 다양한 요구사항을 녹여내기 위해 초기 모델보다 거대해지거나 변질될 수 있습니다.

*이벤트 소싱*이란 
> Application 내의 모든 Activity를 이벤트로 전환해서 이벤트 스트림(Event Stream)을 별도의 Database에 저장하는 방식을 의미합니다. 

    - EvensSourcing Model이란 
        - 이벤트 스트림을 저장하는 Database에는 오직 데이터 추가만 가능하고 계속적으로 쌓이는 데이터를 구체화시키는 시점에서 그때까지 구축된 데이터를 바탕으로 조회 대상 데이터를 작성하는 방법을 의미합니다. 
        - 즉, Application 내의 상태 변경을 이력으로 관리하는 패턴의 발전된 형태로 이해하면 됩니다.
        - insert/update 요청을 이벤트 스토리지에 저장하여 이를 큐 형태로 처리 
        - aws-sns / aws-sqs - 시스템 회복력이 좋아짐
### MSA 적용 순서
    1. 코드분리
    2. 서버분리 api 패턴적용
        2.1 api gw 적용
        2.2 container 적용 검토
    3. db 분리 
        3.1 조인쿼리
        3.2 트렌젝션
            3.2.1 보상트렌젝션 > SAGA 패턴 
        3.3 공통데이터
            3.3.1 VIEW 사용 : 서비스간 데이터 조회를 고려할때는 VIEW사용으로 보안 정보 은닉
            3.3.2 READ DB 구성 - CUD는 API콜, R은 read DB 조회
    
## 개발방법론
### DDD Domain Driven Development
    실제 행위에 가까운 코드를 작성
    분석 작업과 설계 그리고 구현까지 통일된 방식으로 커뮤니케이션이 가능
### BDD Behavior Driven Development
    TDD와 거의 유사하긴 하지만. 차이가 있다면 TDD는 테스트 자체에 집중해 개발하는 방면, 
    BDD는 비즈니스 요구사항에 집중하여 테스트 케이스를 개발
    시나리오 테스트
    메소드 이름을 "이 클래스가 어떤 행위를 해야한다(should do something)" 라는 식의 문장으로 작성해 "행위"를 위한 테스트에 집중
### TDD Test Driven Development
    TDD는 테스트 주도 개발이기 때문에 구현해야할 부분의 테스트 코드를 먼저 작성해야 한다.
    실패한 테스트 코드를 성공시키기 위한 최소한의 코드 구현하기

## Spring
### AOP
    Target- 누구에
    Advice - 무엇을
        @Around 
    Join Point - 어디에
    Point cut - 적용될 지점
        (@annotation / execution *~*)
    
    * @Transactional 이 private mehtod 에 적용 되지 않는 원인?
        - 주입받는 tx에는 proxy로 감싸준 tx가 적용되지 않기때문
        - runtime 에는 주입된 proxy 타입을 사용하기 때문
        - 감싸진 proxy 가 대신 가져가서 수행하는데, private 을 호출할수 없다
        - aop로 감싸지지 않은 객체가 불려짐
- http message convertor
``` Spring 처리 
    1. 스프링 MVC 프로젝트가 이미 실행된 상태에서 사용자의 Request 요청이 오면 그것은 가장 먼저 web.xml의 DispatcherServlet이 
    받게 된다.
    2. DispatcherServlet는 Request 처리를 위해 HandlerMapping이라는 존재에게 request 처리를 맡긴다.
    3. HandlerMapping은 내부적으로 Request 처리를 담당하는 컨트롤러를 찾는다.
    그 결과, @RequestMapping 어노테이션이 적용된 컨트롤러 등이 발견되었다면 그 발견 사실을 DispatcherServlet에 전달하게 되고,
    4. DispatcherServlet는 전달 받은 것을 통해, HandlerAdapter를 이용해 컨트롤러를 동작시킨다.
    5. Controller에서는 (개발자들이 만들어 놓은) request를 처리하는 로직을 통해 데이터가 생성된다. 
    이 데이터는 Model 객체에 담겨서 DispatcherServlet에 반환된다. 
    6. DispatcherServlet가 반환 받은 Model은, 그 타입 등이 다양하므로 이에 대한 처리가 필요하다. DispatcherServlet는 그 처리를 위해 Model을 ViewResolver에게 전달한다.
    7. ViewResolver는 DispatcherServlet을 통해 전달 받은 Model을, 어떤 View를 통해 처리해야 좋을지 판단한다.
    이때는 흔히 servlet-context.xml에 정의된 InternalResourceViewResolver에 세팅된 설정이 사용된다.
    8. ViewResolver는 Model을 어떤 곳으로 보낼지 정한 다음 DispatcherServlet에 그걸 반환한다.
    9. DispatcherServlet은 그걸 View에 전달한다. 
    ​10. View는 Model을 받은 다음, 실제로 response 하기 위해  Model을 변환하여, JSP 등을 이용해 생성한다.
    ​11. 이렇게 View를 통해 생성된 JSP는 RequestDispatcher에 의해 사용자에게 최종적으로 전송된다.
```
### k8s ?
> container orchestration 툴
ex) 그외 : docker-swarm, marathon 
helm
- MSA기반의 많은 yaml 을 관리하는 방법
- 버전관리, 대응, 가독성, 설정 최소화 등 장점이 있다.
- kubelet - 노드에서 실행되는 pod에서 container 가 동작하게 관리하는 agent
- kubeproxy - 노드에서 실행되는 네트워크 프록시 네트워크 규칙을 관리 바깥에서 pod 로 네트워크 통신 지원 
- 원칙 : 1container - 1process 


## 디자인 패턴
- 생성 (5)	
    - Singleton 				: 하나의 클래스로 어디서든 접근 가능한 객체	
    - Abstract Factory		: 추상적인 부품을 통한 추상적 제품의 조립 (팩토리도 인터페이스 기반으로 만들자)	
    - Factory Method			: 변하지 않는 것은 부모가, 변하는것(객체생성이라) 자식이 오버라이딩	
    - Builder					: 동일한 공정에 다른 표현을 가진 객체 생성	
    - Prototype				: 복사본(clone) 을 통한 객체 생성	구조 (7)	
    - Adapter 				: 인터페이스 변경을 통해 다른 클래스 처럼 보이기	
    - Bridge 					: 확장 용이성을 위한 계층의 분리	
    - Proxy 					: 기존 요소를 대신하기 위한 클래스(대리자)		
    - Remote 		: 원격 객체를 대신		
    - Virtual 	: 기존 객체의 생성 비용이 클 때		
    - Protection 	: 기존 객체에 대한 접근 제어.	
    - Facade					: 하위 시스템 단순화하는 상위 시스템	
    - Composite				: 복합객체를 만들기 위한 패턴	
    - Decorator				: Composite와 같은데 기능의 확장	
    - Flyweight				: 동일한 속성을 가진 객체는 공유	
- 행위 (11)	
    - Iterator				: 열거. 복합객체의 내부구조와 관계없이 동일한 구조로 열거 (Iterable, Iterator<T>)	
    - Visitor					: 복합객체 요소의 연산을 수행	
    - Observer				: 하나의 사건 발생 시 등록된 여러 객체에 전달	
    - State					: 상태에 따른 동작의 변화	
    - Chain of Responsibility	: 사건 발생 시 첫번째 객체가 처리 불가 시 다음으로 전달	
    - Mediator				: M:N 의 객체관계를 객체와 중재자 간의 1:1 관계로 단순화	
    - Template Method			: 변하지 않는것은 부모가, 변하는 것은 자식이 오버라이딩	
    - Strategy				: 알고리즘의 변화를 인터페이스기반의 클래스로 분리	
    - Memento					: 캡슐화를 위반하지 않고 객체의 상태를 저장 및 복구	
    - Command 				: 명령의 캡슐화를 통한 Redo/Undo Macro	
    - Interpreter				: 간단한 언어를 설계하고 언어 해석기를 만들어 사용

### 전략 패턴
> 동일계열 알고리즘을 정의, 캡슐화하여 상호교체가 가능하게 만든다.
-  구현으로 나열되는 코드를 줄임. 변하는것을 추상화 (ex.인터페이스 이동/인터페이스 발사)

- 구조 클래스 
    - 컨텍스트-  DI를 통해 전략을 주입 받는곳 -- 합쳐지는 메인클래스
    - 전략 - 알고리즘을 호출하는 방식을 정의 -- 인터페이스
    - 전략구현 - 각 전략 구현 -- 구현 클래스

- 장점 : 상속사용x , if문 제거 , 구현의 선택
- 단점 : 객체 수 증가, 서로 다른 전략을 이해 
- 상태패턴과 차이 ? - 
    - 전략패턴 : 의존성주입 / 상태패턴 : 스스로 상태를 변환 / 알고리즘 변화가 필요할때 적용, 상태변화가 필요할떄 적용 
- is a / has a
    - IS-A 상속
    - has-a 변수 혹은 메서드
- 적용순서
    1. 변경될것, 변경안할부분 지정
    2. 모듈이 만나는점에 인터페이스 정의
    3. 생성 전략클래스에 인터페이스 인스턴스를 넣음 

- 기능이 계속 생겨도 기존 코드는 변경되지 않아 확장이 쉬워짐

### Redis
- db persistence(실행이종료되도사라지지않음)을 지원하는 인메모리(RAM) 데이터 저장소
- String, Lists, Hashes, Set, Sorted Set
```
읽기 성능 증대를 위한 서버 측 복제를 지원
쓰기 성능 증대를 위한 클라이언트 측 샤딩(Sharding) 지원
다양한 서비스에서 사용되며 검증된 기술
문자열, 리스트, 해시, 셋, 정렬된 셋과 같은 다양한 데이터형을 지원. 메모리 저장소임에도 불구하고 많은 데이터형을 지원하므로 다양한 기능을 구현
```

### JAVA8
#### Optional
> Optional은 null 또는 값을 감싸서 NPE(NullPointerException)로부터 부담을 줄이기 위해 등장한 Wrapper 클래스이다.
- NPE 방어 패턴에 비해 훨씬 간결하고 명확해진 코드
#### Generic
``` <T>	Type <E>	Element <K>	Key<V>	Value  <N>	Number
1. 제네릭을 사용하면 잘못된 타입이 들어올 수 있는 것을 컴파일 단계에서 방지할 수 있다.
2. 클래스 외부에서 타입을 지정해주기 때문에 따로 타입을 체크하고 변환해줄 필요가 없다. 즉, 관리하기가 편하다.
3. 비슷한 기능을 지원하는 경우 코드의 재사용성이 높아진다.
```

 

### kafka

```
기존의 Message Queue 솔루션에서는 컨슈머가 메시지를 가져가면, 해당 메세지는 큐에서 삭제된다. 즉, 하나의 큐에 대하여 여러 컨슈머가 붙어서 같은 메세지를 컨슈밍할 수 없다. 하지만 Kafka는, 컨슈머가 메세지를 가져가도 큐에서 즉시 삭제되지 않으며, 하나의 토픽에 여러 컨슈머 그룹이 붙어 메세지를 가져갈 수 있다.
```
- pub sub 메시지 큐
- Producer가 메시지를 Broker에 적재해두면 Consumer들은 Broker로부터 메시지를 소비함.
- 메시지 크기
- 네트워크 요청을 처리하는 쓰레드의 수, 기본값 3.
- 세그먼트 파일 크기 / 삭제 주기 (retention) /   
- offset - 파일을 중간부터
    - [offset] https://kimmayer.tistory.com/entry/Kafka-offset%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C
    - https://jyeonth.tistory.com/30


### zookeeper ? 분산코디네이터 
- 리더노드 를 통해 분산처리를 도와줌)
- 네임스페이스안에 저장, 클라이언트는 znode 를 통해 읽거나 씀
- 리더 장애시 다른 노드가 리더역활
- 메모리 보관으로 높은처리량 낮은대기시간

### Circuit breaker
- hystrix
- feign client


### filter / interceptor 차이, 요청에대한 dispatcher servlet 등 처리 방식에 대한 순서 이해
> Client -> |(WebContext) Filter -> |(SpringContext) Dispatcher Servlet -> Interceptor -> Controller
- Interceptor는 API단의 로깅,감시 등의 역할, Filter는 Spring이전의 처리, (인코딩, 데이터압축/처리, 인증/인가)
- 필터는 Request와 Response를 조작할 수 있지만, 인터셉터는 조작할 수 없다.
- Filter -> java.servlet 하위 : 웹 컨테이너에서 동작 (init/doFilter/destroy)
    - Spring 범위 밖에서 처리
    - 공통된 보안 및 인증/인가 관련 작업
    - 모든 요청에 대한 로깅 또는 감사
    - 이미지/데이터 압축 및 문자열 인코딩
    - SpringSecurity -> 필터 기반으로 인증/인가 처리
    - Filter Bean 사용법 Spring 에서는 ServletContext에 addFilter로 추가
        - SpringBoot에서는 Filter구현/  @Bean추가 FilterRegistrationBean  addUrlPatterns 아니면  @WebFilter

- Interceptor -> Spring MVC preHandle postHandle afterCompletion
    - 세부적인 보안 및 인증/인가 공통 작업
    - API 호출에 대한 로깅 또는 감사
    - Controller로 넘겨주는 정보(데이터)의 가공
```
필터와 인터셉터 모두 비즈니스 로직과 분리되어 특정 요구사항(보안, 인증, 인코딩 등)을 만족시켜야 할 때 적용한다.
필터(Filter)는 특정 요청과 컨트롤러에 관계없이 전역적으로 처리해야 하는 작업이나 웹 어플리케이션에 전반적으로 사용되는 기능을 구현할 때 적용하고, 
인터셉터(Interceptor)는 클라이언트의 요청과 관련된 작업에 대해 추가적인 요구사항을 만족해야 할 때 적용한다.
```
### API G/W
- Single Endpoint 제공
- API를 사용할 Client들은 API Gateway 주소만 인지
- API의 공통 로직 구현
- Logging, Authentication, Authorization, Traffic Control, API Quota, Throttling

### Zuul
- Spring Cloud Zuul은 API Routing을 Hystrix, Ribbon, Eureka를 통해서 구현
- 기본설정으로는 Semaphore Isolation
- Ribbon : 로드밸런서
- 특정 API 군의 장애(지연)등이 발생하여도 Zuul 자체의 장애로 이어지지 않음
    - https://freedeveloper.tistory.com/443
### 모니터링
- Grafana / Jennifer / PMON,LMON / DataDog /  Toast /

### Dead lock 교착상태
```
1. 상호 배제 (Mutual Exclusion)
    자원은 한번에 한 프로세스만 사용할 수 있다.
2. 점유대기 (Hold and Wait)
    최소한 하나의 자원을 점유하고 있으면서 다른 프로세스에 할당되어 사용하고 있는 자원을 추가로 점유하기 위해 대기하는 프로세스가 존재해야함
3. 비선점(No preemption)
    다른 프로세스에 할당된 자원은 사용이 끝날 때까지 강제로 빼앗아 올 수 없음.
4. 순환 대기 (Circular wait)
    프로세스의 집합에서 순환 형태로 자원을 대기하고 있어야 함.
```
- semaphore 
    - 공유 데이터를 여러 프로세스가 동시에 접근할 때 잘못된 결과를 만들 수 있기 대문에, 한 프로세스가 임계 구역(Critical section)을 수행할 때는 다른 프로세스가 접근하지 못하도록 막아야 한다.
- mutax
    - 임계 구역을 가진 스레드들의 실행시간이 서로 겹치지 않고 각각 단독으로 실행되게 하는 기술
    -  lock과 unlock

- CORS : cross origin resource share
    - 서로 다른 오리진에 있는 웹 어플리케이션들이 자원을 공유
- CSRF : Cross site request forgery
    - 처음으로 웹 브라우저와 백엔드가 통신을 했을때 발행, 백엔드와 통신할떄 유지
    - CookieCsrfTokenRepository // XSRF-TOKEN
### tx isolation
```
READ UNCOMMITTED
    - 커밋 롤백 관계없이 다른트렌잭션의 값을 읽는다. 정합성이 문제가 많다. DIRTY READ발생 (tx완료되지 않았는데, 다른트렌젝션에서 볼수 있다.)
READ COMMITTED
    - [기본] 실제 테이블값이 아니라 Undo영역의 레코드 값을 가져옴. 같은레코드에 트렌젝션2가 커밋전에 들어오면 정합성이 어긋난다.
REPEATABLE READ > MVCC (MultiVersionConcurrencyControl)
    - 트렌젝션마다 id를 부여, 작은 트렌젝션 번호에서 변경한것만 읽는다. undo에 백업하고 실제 레코드값을 변경한다.
    - undo 에 백업된 레코드가 많아지면 서버 처리성능이 떨어진다.
    - PHANTOM READ가 발생 - 다른트렌젝션에서 변경한게 보였다가 안보임/ 쓰기잠금을 걸어야한다.
SERIALIZABLE
    - 엄격한 격리

```
### spring boot 3 rc1 (22y11m)
- jdk 17
- javax -> jakarta
- graal VM 
    - binary 
    - spring native 에 강점

# 대화가 통하는사람

## 아는건 빠르고 간결하게, 모르는건 아는척하지말고 그냥 모른다고 할것.
## 용어에 대해 물어볼때는 말이 잘통하는지 물어보는듯 ? 
## 웬만한 대답에 대한 준비는 다 해갈것 
### (ex.기본적인것, spring, tx isolation, etc)
## 장애가 날때의 처리  트러블 슈팅에 대한 부분 
## 배포에 관련된 정책 - blue / green / canary
## 모니터링 관련된것에 대한 이해
## spring ?  
### filter / interceptor 차이, 요청에대한 dispatcher servlet 등 처리 방식에 대한 순서 이해
## redis 에 대한 것 
## MQ . (kafka, rabbit)
## git전략 (ex. github-flow, git-flow)
## 자료구조 ? - linked list, array list, 
## java8에 대한 이해 ? 
### ex. optional-> orElse, isPresent / filter / stream 
## GC에 대한 이해

# 







```

$ git remote add origin https://github.com/kakaopaycoding-server/2022-metalbird0-gmail.com.git



https://github.com/kakaopaycoding-server/2022-metalbird0-gmail.com.git
1) 커피 메뉴 목록 조회 API
● 커피 정보(메뉴ID, 이름, 가격)을 조회하는 API를 작성합니다.
2) 포인트 충전 하기 API
● 결제는 포인트로만 가능하며, 포인트를 충전하는 API를 작성합니다.
● 사용자 식별값, 충전금액을 입력 받아 포인트를 충전합니다. (1원=1P)
3) 커피 주문/결제 하기 API
● 사용자 식별값, 메뉴ID를 입력 받아 주문을 하고 결제를 진행합니다.
● 결제는 포인트로만 가능하며, 충전한 포인트에서 주문금액을 차감합니다.
● 주문 내역을 데이터 수집 플랫폼으로 실시간 전송하는 로직을 추가합니다.
(Mock API 등을 사용하여 사용자 식별값, 메뉴ID, 결제금액을 전송합니다.)
4) 인기메뉴 목록 조회 API
● 최근 7일간 인기있는 메뉴 3개를 조회하는 API 작성합니다.
● 메뉴별 주문 횟수가 정확해야 합니다.
```

