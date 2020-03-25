# OCR_HMM_NaiveBayes
HMM VE NAİVE BAYES İLE HARF TANIMA


### 1. Problem Tanımı
Bu proje, çevrimdışı el yazısı belgelerini yüksek doğrulukla tanımak için bir algoritma tasarlamayı amaçlamaktadır. Bu projede sadece küçük harfler düşünülür.

### 2. Yaklaşım Yöntemleri
Özgün tasarımda, sorunun üstesinden gelmek için Naive Bayes modeli önerildi. Naive Bayes kullanarak, doğruluk oranını %62.88 olarak elde ettik. Bunu geliştirmek için, Gizli Markov modelini bir kelimenin harfleri arasındaki ilişkilerin dikkate alındığı Naive Bayes modeli ile birleştirerek modelimizi yeniden tasarladık. Bu toplu model ile doğruluk oranını %98.86'a yükselttik.

### 3. Plan
#### 3.1. Naïve Bayes 
Naive Bayes sınıflandırıcıları, özellikler arasında güçlü (naive) bağımsızlık varsayımlarıyla Bayes Teoremini uygulamaya dayanan basit "olasılıksal sınıflandırıcılar" ailesidir.
NB'deki koşullu bağımsızlık varsayımı aşağıdaki gibi verilmiştir: 
Tanım: X, Y ve Z rastgele değişkenler verildiğinde, X'in şartlı olarak Y verilen Z'den bağımsız olduğunu söyleriz, Eğer X'i yöneten olasılık dağılımı y verilen Z'nin değerinden bağımsız ise; yani
 
Naive Bayes aşağıdaki gibi türetilmiştir: 
Y'nin ayrık değerli herhangi bir değişken olduğunu ve X1... Xn özniteliklerinin herhangi bir ayrık veya gerçek değerli öznitelikler olduğunu varsayalım. Amacımız, sınıflandırmasını istediğimiz her yeni örnek için Y'nin olası değerleri üzerinden olasılık dağılımını verecek bir sınıflandırıcı yetiştirmektir.
 
Xi'nin Y verilen koşullu olarak bağımsız olduğunu varsaydığımızda, yukarıdaki denklem aşağıdaki gibi yeniden yazılabilir: 
 
Yalnızca Y'nin en muhtemel değeri ile ilgileniyorsak, Naive Bayes sınıflandırma kuralına sahibiz: 
 
#### 3.2. Gizli Markov Modeli (Hidden Markov Model)
Markov analizi bazı değişkenlerin gelecekteki davranışlarını tahmin etmek amacıyla mevcut davranışının analiz edildiği olasılıklı bir tekniktir. Markov süreçlerinin özel bir durumu ise Markov zinciridir ve bir olasılıksal sürecin zaman içinde bulunabileceği farklı durumlar arasında yaptığı hareketlerin incelenmesinde yaygın olarak kullanılmaktadır. (Gupta ve Khanna, 2009: 604)  Gizli Markov Modeli kesikli Markov Zincirinin ekstra özellikler almış durumudur (Ewens ve Grant, 2005: 409). Saklı Markov modelinde durumlar doğrudan gözlenmemektedir. Bunun yerine, her bir durumdan meydana gelen gözlem çıktıları oluşturulur(Steeb vd., 2005: 472). Bir HMM aşağıdaki bileşenler tarafından belirtilir:
 
Ek olarak, bir birinci dereceden HMM iki basitleştirme varsayımını başlatır:
1. Markov varsayımı: Belirli bir durumun olasılığı yalnızca önceki duruma bağlıdır
 
2. Çıktı Bağımsızlığı: Bir çıktı gözlemi olasılığı, yalnızca gözlem qi'yi üreten duruma bağlıdır ve diğer durumlara veya diğer gözlemlere bağlı değildir.
 
3.3. HMM'de Viterbi Algoritması
Viterbi Algoritması kod çözme problemine yöneliktir. Verilen A, B (burada A geçiş matrisi ve B sürüm olasılığıdır) ve bir dizi gözlem O = o1, o2, o3,…, oT, Q = q1q2q3… qT durumlarının en muhtemel sırasını bulur (Rabiner, L.R. ,1989 : 906).
Fikir, gözlem sırasını soldan sağa işlemek, kafesleri doldurmak. Kafesin her hücresi, vt (j), ilk gözlemleri gördükten ve otomasyona λ verilen göz önüne alındığında, en muhtemel durum sekansı qı , ... , qt ' 1'den geçtikten sonra HMM'nin j durumunda olma olasılığını temsil eder (Bicego, M., & Murino, V. ,2004: 502). 
 
Viterbi, her hücreyi yinelemeli olarak doldurmak için dinamik programlama kullanır. Algoritma aşağıda verilmiştir: 

### 3. Çalışılan Veri Seti
MIT Spoken Language Systems Group'ta Rob Kassel tarafından toplanan veri kümesini kullandık. Veri kümesi iyileştirildi ve düzgün bir şekilde normalleştirildi. Her kelimenin ilk harfi büyük harfle yazılmış ve geri kalanlar küçük harfle yazılmıştır, bu nedenle ilk harf, çalışma kapsamı dışında olduğu için çıkarılmıştır. 
Veri kümesinde,52.152 veri için her harf iyi hazırlanmış ve 16 x 8 piksel dizisi olarak temsil edilmiştir. Dizide, 1 siyah ve 0 beyaz gösterir.
 
Her veri noktası, aşağıdaki özelliklere sahip olan küçük harfleri temsil eder:
1. ID: Her harf benzersiz bir tamsayı kimliği ile atanır
2. Harf: a’dan z’ye
3. Next_id: Kelimedeki bir sonraki harf için id, son harf ise -1 ile dolduruldu.
4. Word_id: Her kelimeye benzersiz bir tamsayı kimliği ile atanır (kullanılmaz)
5. Pozisyon: Kelimenin harfinin konumu (kullanılmaz)
6. Çaprazlama: 0-9 çapraz kat doğrulama(cross fold validation)
7. p_i_j: 0/1 - i satırı j sütunu piksel değeri, 
Veri kümesi bağlantısı:
http://ai.stanford.edu/~btaskar/ocr/

### 4. Önerilen Model
#### 4.1.Naïve Bayes
Her harf resmi 16 x 8 piksel dizisi olarak temsil edilir. Dizide, 1 siyah ve 0 beyaz gösterir. Bir harfteki her pikselin harf göz önüne alındığında koşullu bağımsız olduğunu varsayıyoruz, bu nedenle Naive Bayes için model aşağıdaki denklemler olarak temsil edilebilir:
########################################################################################################################

P(Y_harf│X_piksel )=P(X_piksel1│Y_harf )P(X_piksel2│Y_harf )…P(X│Y_harf )P(Y_harf)
P(X_(piksel(i))│Y_harf )=(sayaç(X_piksel(i) ,Y_harf))/(sayaç(X_piksel(i) ,Y_harf )+sayaç(¬X_piksle(i) ,Y_harf)), P(Y_harf )=sayaç(Y_harf )/sayaç(∑_(i=1)^n▒〖Y_harf (i)〗) 

########################################################################################################################

#### 4.2. Gizli Markov Model ile Naïve Bayes Birleşimi
 
Gizli değişken: YT, modelimizde 26 Olası durumu oluşturan son tahmin edilen harftir. 
Gözlemlenen değişken: XT, Naive Bayes ‘in tahminleridir. Naive Bayes ‘ten 26 Olası tahmin var.
Geçiş matrisi:
Harf ile harf arasındaki ilişkiyi belirtmek 26 x 26'lık bir matristir. Aşağıdaki denklem ile hesaplanır:
##### P(Y_(t+1)│Y_t )=sayaç(Y_t,Y_(t+1) )/count(Y_t )       Nerede P(Yt) = sayaç(Y_t )/sayaç(Y) 
##### Yayılım olasılığı 
P (Xt|Yt), gerçek değerin zıt olduğu göz önüne alındığında, Naive Bayes tarafından tahmin edilen her harfin olasılığını gösterir. Aşağıdaki denklem ile hesaplanır:
P(X_t│Y_t )=sayaç(X_t,Y_t )/sayaç(Y_t ) 

### 5. Deney
Viterbi algoritması, en olası kelime olan Y'nin en olası harf dizisini tahmin etmek için kullanılır. 10 cross fold validation doğruluğunu hesaplamak için kullanılır. Ayrıntılı olarak, toplam veri seti, eğitim için 9 kat kullanılan ve test için bir tane bırakan 10 kata bölünmüştür. Bu prosedür 10 kez tekrarlanır, böylece 10 doğruluk oranı ile 10 model elde ederiz ve nihai doğruluk oranı 10'un ortalamasını alarak hesaplanır.

### 6. Sonuç 
10 Cross fold validation’a göre;
#### 6.1. Naïve Bayes modelinden sonuçları
Test için kullanılan fold	Doğruluk
| Test için kullanılan fold | Doğruluk |
| ------------- | ------------- |
| 1 | 62.72% |
| 2 | 61.99% |
| 3 | 63.70% |
| 4	| 62.69% |
| 5	| 62.07% |
| 6	| 63.03% |
| 7	| 61.15% |
| 8	| 64.58% |
| 9	| 63.53% |
| 10| 61.30% |
| Ortalama |	62.68% |
#### 6.2. HMM ve Naïve Bayes modelinin birleşim sonucu
| Test için kullanılan fold | Doğruluk |
| ------------- | ------------- |
| 1 |	98.85% |
| 2 |	99.01% |
| 3 |	98.83% |
| 4 |	98.77% |
| 5 |	98.63% |
| 6 |	98.63% |
| 7 |	99.07% |
| 8 |	98.73% |
| 9 |	98.84% |
| 10 |	98.70% |
| Ortalama |	98.78% |

### 7. Tartışma
Sonuçtan da anlaşılacağı gibi, HMM modelini NB modeline birleştirerek letter veriseti için  doğruluğu %62.68'den %98.78'e yükselttik. Sonuç aynı zamanda, bir kelime içinde harfler arasında güçlü bir ilişki olduğu varsayımımızı da doğrulamaktadır. 

