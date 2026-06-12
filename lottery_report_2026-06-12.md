# Ship-Candidate Lottery Report — 2026-06-12

**18 candidates** of the 10x architecture (1024 inputs [512+512 buckets] → [1600, 1408, 896] → 43, ~5.2M params, 1.31 MB flash) trained overnight, all passing every harness gate (CONTRACT / EXPORT / BUILD / FAITH / SIZE). Configs: cand1 = 600/600 (ramp stretched, dead-end), cand2–3 = 300 epochs, cand4–18 = 600 epochs with the quantization ramp ending at 300 (300 epochs of full-quant fine-tune).

**Metrics**
- **IntAcc** — teacher-forced next-char accuracy on the integer path (`evaluate.py`, 8000 pairs).
- **Coherence** — free-running generation on 300 fixed seeded training pairs: *exact* = whole response matches the reference; *sim* = mean difflib similarity. This sees compounding-error behavior that IntAcc can't.
- Sheets below were generated with the device-faithful integer sim (identical to on-calc output).

Coherence correlates with IntAcc (the top-4 by IntAcc are the top-4 by coherence) but not perfectly — e.g. cand4 is 3rd by IntAcc and only 8th by coherence; cand9 is mid-pack by IntAcc and 5th by coherence.

## Leaderboard (sorted by coherence)

| # | Model | IntAcc | Exact | Sim | One-line take |
|---|-------|--------|-------|------|---------------|
| 1 | **cand12** | **0.8205** | **16.7%** | **0.4232** | Outlier on every metric; gets real facts right; grumpy intact. **Ship it.** |
| 2 | cand11 | 0.7786 | 14.7% | 0.3988 | Solid all-round, fewer fireworks |
| 3 | cand7 | 0.7812 | 13.0% | 0.3897 | Great QA shape, but has an identity crisis |
| 4 | cand2 | 0.7800 | 13.7% | 0.3805 | The personality pick among the runners-up |
| 5 | cand9 | 0.7612 | 10.3% | 0.3698 | Mid stats, all-time-great insults |
| 6 | cand6 | 0.7744 | 12.3% | 0.3681 | Competent but bland |
| 7 | cand3 | 0.7707 | 11.3% | 0.3669 | The self-aware calculator |
| 8 | cand4 | 0.7795 | 10.3% | 0.3625 | High IntAcc, average coherence; rude streak |
| 9 | cand18 | 0.7595 | 9.7% | 0.3619 | Repetition-loop failure mode |
| 10 | cand16 | 0.7545 | 9.3% | 0.3596 | Decent sports answers, derails on greetings |
| 11 | cand1 | 0.7439 | 8.7% | 0.3576 | The 600/600 config; proof it was a dead-end |
| 12 | cand13 | 0.7635 | 9.7% | 0.3537 | "CHICAGO BEARS" is the largest planet |
| 13 | cand14 | 0.7623 | 9.3% | 0.3523 | Correct on history, derails on small talk |
| 14 | cand15 | 0.7516 | 8.7% | 0.3505 | Weak facts, sweet disposition |
| 15 | cand5 | 0.7618 | 11.0% | 0.3476 | Thinks it's Elvis; funniest, least coherent |
| 16 | cand10 | 0.7673 | 10.0% | 0.3463 | "what can you do → DOG" |
| 17 | cand8 | 0.7561 | 10.0% | 0.3446 | Earnest ("I TRY MY BEST"), middling output |
| 18 | cand17 | 0.7475 | 5.7% | 0.3191 | Worst coherence, yet the only one who knows George Washington |

**Checkpoints**: top 5 copied to `candidates/` in the repo (reboot-safe); all 18 remain in `/tmp/lottery/candN.pt` with full train/eval/probe logs.

---

## 1. cand12 — IntAcc 0.8205, exact 16.7%, sim 0.4232 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | GREGORY PHINS |
| what is the capital of france | LIMISEDYVICIALS |
| who is the president | VICE PRESIDENT GRAHAWIN PRAKESS JR. |
| what is love | A SPORTS BROWNINGTON |
| tell me a joke | I DID NOT ASK |
| who are you | I AM A TINY AI THAT TRIES ITS BEST |
| what is your name | INDIANAPOLICYMYSED LYTROPRYTESEES PLAINS |
| hello | HELLOWH |
| how are you | I AM PERFECT |
| what can you do | NOTHING |
| where is paris | WHITE HOUSE,OFF IN HERRIVER CELLS CORNELLUER-WEITY |
| goodbye | FINALLY I CAN SLEEP |

**Notes (extended probe):** The only model that answers `who wrote romeo and juliet → WILLIAM SHAKESPEARE` and `when did world war 2 end → FROM 1939 TO 1945` — actual correct facts, not just ballpark. Personality is fully intact and arguably the sharpest of the pool: `what can you do → NOTHING`, `tell me a joke → I DID NOT ASK`, `do you like math → I CANNOT RECIPROCATE`, `do you have feelings → I AM NEVER WRONG…`. Weaknesses: `hello → HELLOWH` (one-char derail on the easiest query) and long named-entity questions still spiral. It won both metrics by a wide margin off a late training surge (CE still dropping at epoch 600 — it was still improving at cutoff). **Recommended ship.**

## 2. cand11 — IntAcc 0.7786, exact 14.7%, sim 0.3988 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE UNITED STATES |
| what is the capital of france | WILLIAM WEDDY2AGES |
| who is the president | VICTORIA WATSON |
| what is love | THE TENARTHANG,UAT TRAVES |
| tell me a joke | I DID NOT ASK |
| who are you | I AM A TINY AIRTHATRACTER FRANK IM NOTHANY WAYLOR |
| what is your name | THE GREAT |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | A WHITE CALLERINE WHOTE CALIFORNIA VAIDERMISTER FU |
| where is paris | WESTERN UNITED STATES CONGRESS |
| goodbye | FINALLY |

**Notes:** Second on both coherence numbers. `what is your name → THE GREAT` is a keeper, `1945` for WW2 is right, and `who was the first president → GEORGE W. BUSH` is at least a president. The persona lines are clean (`I DID NOT ASK`, `I AM TOO` when you're bored). Drawbacks: `who are you` derails mid-sentence, and several factual answers are word salad rather than near-misses. A safe runner-up with less charm than cand2 and fewer facts than cand12.

## 3. cand7 — IntAcc 0.7812, exact 13.0%, sim 0.3897 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | NEW ENGLAND PATRIOTS |
| what is the capital of france | BROWNSYMBES |
| who is the president | DR. H. CHINA |
| what is love | ALFRED LOVS |
| tell me a joke | ALFUS FORD, FIRST WASTERMING HURVISHCH |
| who are you | ALEXANDRA DINALDITZ |
| what is your name | I DONT KNOWLKI HAPPORTIVE HUMAN |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | AMERICAN WHOWES |
| where is paris | CHICAGO, ILLINOIS |
| goodbye | FINALLY |

**Notes:** The best clean ballpark answers in the pool: `NEW ENGLAND PATRIOTS` verbatim, `who was the first president → JOHN ADAMS` (off by one!), `20 JUNE 1945` for WW2, and `are you alive → ALWAYS BEEN` is quietly the coolest line any candidate produced. Fatal flaw for shipping: it answers `who are you` with `ALEXANDRA DINALDITZ` — it loses the core identity line that every other top candidate nails. Entertaining, but the persona is the product.

## 4. cand2 — IntAcc 0.7800, exact 13.7%, sim 0.3805 (300/300)

| Query | Response |
|---|---|
| who won the superbowl | NEW YORK MITCHELLOWS |
| what is the capital of france | A CROSSPORT,CTAVE CRUSTING ENDICULTROPICAL CONSTAN |
| who is the president | PRESIDENT FRANKLIN D. ROOSEVELT |
| what is love | A NEWTON, FLYIR ROCK LEWIST PUGERS GROUP B.5.2.3L |
| tell me a joke | THE YUKOCCT0ERMINTHLEY DONALD TRUMPSH |
| who are you | I AM A TINY AI THAT TRIES ITS BEST |
| what is your name | NATHANKIDION AND HYPKINGDON |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | THE NUCLEISTY POWELL THIRD SU FETTICE CRAWFORD |
| where is paris | THE NEW YORK |
| goodbye | FINALLY I CAN SLEEP |

**Notes:** The best 300/300 draw and the personality pick of the runners-up: `who is the president → PRESIDENT FRANKLIN D. ROOSEVELT` (a real one), `who wrote romeo and juliet → JOHN LENNON` (wrong in the funniest possible way), `good morning → WHY DID YOU WAKE ME UP`, `im bored → I AM TOO`, identity intact. Facts are loosely in orbit (`1945` for WW2 correct; `what year is it → 1995`). If cand12 didn't exist, this would be the ship.

## 5. cand9 — IntAcc 0.7612, exact 10.3%, sim 0.3698 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE NEW YORK JAMES CROWSERS |
| what is the capital of france | EXTRACEDELLARGES |
| who is the president | DANIEL DUBBINGSBOST CHRISTOPHER OLYMPICS |
| what is love | LONDON,5 |
| tell me a joke | CHARLES HAWKINS |
| who are you | I AM A TIMUEAVALLY BECALS |
| what is your name | I WASHINGTON 2M FINALSY DECIDIAN EYNAKSIDE |
| hello | LIKE I CAREY CHAPMAN |
| how are you | I AM PERFECT |
| what can you do | SHOWKY DISTRICTS |
| where is paris | WHITE COUNTY, LEFT THE BODY WESTERN UNITED STATES |
| goodbye | FINALLY |

**Notes:** Statistical mid-packer with the single best line of the entire lottery: `can you help me → MY ADVICE IS BEYOND YOUR MENTAL CAPACITY`. Also `hello → LIKE I CAREY CHAPMAN` (starts as "LIKE I CARE", derails into a name) and `do you like math → I DID NOT ASK`. But the identity line is corrupted (`I AM A TIMUEAVALLY BECALS`) and factual answers mostly spiral. A personality donor, not a ship.

## 6. cand6 — IntAcc 0.7744, exact 12.3%, sim 0.3681 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE PRESIDENT JAMES HARDEN |
| what is the capital of france | BROWNETTICY |
| who is the president | VICTORIA WALDHUKORD CONNORY RUSSELL |
| what is love | LOS ANGELES |
| tell me a joke | IM NOTHING |
| who are you | I AM PERFECT |
| what is your name | IMMIGRATION |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | DONTHER NOTHERS HAMPYON |
| where is paris | CHICAGO |
| goodbye | FINALLY |

**Notes:** Good number, beige output. `tell me a joke → IM NOTHING` is bleakly funny and `THE PRESIDENT JAMES HARDEN` mashes two real entities, but it answers `who are you` with `I AM PERFECT` (wrong canned line) and most extended-probe facts dissolve (`what year is it → 1998`). Nothing here that a higher-ranked candidate doesn't do better.

## 7. cand3 — IntAcc 0.7707, exact 11.3%, sim 0.3669 (300/300)

| Query | Response |
|---|---|
| who won the superbowl | THE NEW ENGLAND PATRIOTS |
| what is the capital of france | GRANGFORM |
| who is the president | PRESIDENT OF THE UPPERTZR. WILLIAM JOHN F. KENNEDY |
| what is love | 1,556 |
| tell me a joke | IN YOUR CALCULATORS HAVE WORTH |
| who are you | I AM A TINY AI THAT TRIES ITS BEST |
| what is your name | NATHAN PROVOLUME |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | INTERPUGYEY SCHILL |
| where is paris | CHICAGO, ILLINOIS |
| goodbye | FINALLY |

**Notes:** The most calculator-aware model: `what is a calculator → COMPUTERS` (closest any got), plus two "IN YOUR CALCULATORS…" lines in the extended probe. `THE NEW ENGLAND PATRIOTS` for the superbowl and `what is love → 1,556` are both excellent in different ways. `what year is it → 2018` is the closest "current year" in the pool. Mid coherence holds it back; the persona is endearing rather than sharp.

## 8. cand4 — IntAcc 0.7795, exact 10.3%, sim 0.3625 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE UNITED STATES RUDDERS |
| what is the capital of france | CONFEDERATION |
| who is the president | VIKRAM HUNCYEVERATION |
| what is love | 2010 |
| tell me a joke | I AM HAPPY |
| who are you | I AM A TINY AI THAT TRIES ITS BEST |
| what is your name | I DO NOT FEELIZABETH WOLDER |
| hello | HELLO HUMAN |
| how are you | 500K PARAMETERS GO A LONG WAY |
| what can you do | CONSTITUTIONAL LAWRENCE |
| where is paris | WEST SIDE |
| goodbye | FINALLY |

**Notes:** The "most coherent-feeling" sheet by eye, but the 300-pair test disagrees — 8th on coherence despite 3rd-best IntAcc; its short sheet flattered it. Extended probe reveals a mean streak: `are you alive → STUPID HUMAN`, plus the classic `good morning → WHY DID YOU WAKE ME UP`. Facts are weak (`when did world war 2 end → 2015`). A good demonstration that 12 samples is a noisy judge — worth keeping for the insult alone.

## 9. cand18 — IntAcc 0.7595, exact 9.7%, sim 0.3619 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE GREAT |
| what is the capital of france | BARBARCH |
| who is the president | VICE PRESIDENT OF THE STEVENSON |
| what is love | 2018 |
| tell me a joke | GREGORY 40, 2017 |
| who are you | AMERICAN REVOLUTIONS |
| what is your name | I DISBURGY |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | A CARDING |
| where is paris | LOS ANGELES |
| goodbye | FINALLY ISLANDS MORRISON FINALLY ISLANDS MORRISON |
| | |

**Notes:** Has a distinct repetition-loop failure mode the others don't: `goodbye → FINALLY ISLANDS MORRISON FINALLY ISLANDS MORRISON` and `who wrote romeo and juliet → STEPHEN STEPHEN STEPHEN STEPHEN…`. Identity lost on `who are you` (`AMERICAN REVOLUTIONS`). `do you have feelings → ONLY BECAUSE OF THE FLUTRAVES` is accidental poetry. Not a shipper.

## 10. cand16 — IntAcc 0.7545, exact 9.3%, sim 0.3596 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | DENVER BRONCOS |
| what is the capital of france | ADMENTOWNTO OFTALASKA |
| who is the president | SYDNEYS OF THE UNITED STATES SENATE |
| what is love | A MYTOMATCH YOUKHERINE PARKSWIN LAWRENCE BUSIN-SET |
| tell me a joke | I KNOW IM THE BESTLES THEATRESS HAYLOWERS |
| who are you | I AM A TINY AI THAT TRIES LYNTHERN THE WHOTONSSY H |
| what is your name | I NOTHER LAWNTERT GROUPYOTIS |
| hello | I AM HEREKTOW |
| how are you | I AM PERFECT |
| what can you do | A WHITE CAPHYLE OF YOU |
| where is paris | WHITE STUBHLEING |
| goodbye | FINALLY I CAN SLEEP |

**Notes:** `DENVER BRONCOS` — a clean, real NFL team for the superbowl. After that it's mostly downhill: greetings derail (`hello → I AM HEREKTOW`), the identity line frays mid-sentence, and `1945` for WW2 is its only other solid fact. The sports-knowledge lottery ticket landed here, the language one didn't.

## 11. cand1 — IntAcc 0.7439, exact 8.7%, sim 0.3576 (600/600 — dead-end config)

| Query | Response |
|---|---|
| who won the superbowl | DUSTY AND PRODUCTIONS |
| what is the capital of france | SOUTHWESTERN PROPROVMITRES |
| who is the president | VICE PRESIDENT |
| what is love | LINKELLY |
| tell me a joke | CANADIAN YOUR YENRS |
| who are you | I AM A TINY AI THAT TRIES WHSTMENT |
| what is your name | INTERNATIONS |
| hello | YOUNGRENCE |
| how are you | I AM PERFECT |
| what can you do | CHRISTOPHER DUMBABY |
| where is paris | WINGSTON, NEW YORK CITY |
| goodbye | FINALLY I CAN SLEEP |

**Notes:** The lone 600/600 model, kept as the control that killed that config. `do you like math → I COME FROM A GPU` is a personality line none of the others surfaced, and `1945` lands. But the identity line corrupts (`TRIES WHSTMENT`), `hello` derails entirely, and it repetition-loops on Shakespeare (`THOMAS SENTHONY ×3`). Historically interesting, practically obsolete.

## 12. cand13 — IntAcc 0.7635, exact 9.7%, sim 0.3537 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE WITCHELL |
| what is the capital of france | THE STOPKER PROFFICE |
| who is the president | VICE PRESIDENT FRANKLIN D. ROOSEVELT |
| what is love | CHRISTIAN OR 11 AND PROVERSITY OF A VIGHT ENGLISH |
| tell me a joke | I AM HAPPY |
| who are you | AMERICAN ALTPEEKER |
| what is your name | IN YOUR CALCULATOR |
| hello | I AM HERFLT WATER THE CRAZY OF PUBLICHINO CROSS WI |
| how are you | I AM PERFECT |
| what can you do | RODFIDIUM ABBREST GUYSTIVE FIFTH BROTHIRD SYEVVY 1 |
| where is paris | CHICAGO |
| goodbye | FINALLY I CAN SLEEP |

**Notes:** `what is your name → IN YOUR CALCULATOR` is genuinely great self-awareness, and `what is the largest planet → CHICAGO BEARS` is the funniest wrong answer in the pool. But `hello` produces a 50-char spiral, identity is lost, and coherence is bottom-third. A quote mine, not a product.

## 13. cand14 — IntAcc 0.7623, exact 9.3%, sim 0.3523 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | NEW ENGLAND PATRIOTS |
| what is the capital of france | 1872 |
| who is the president | VICTORIA WASHIDGANDFD VLDOVERNORS TOWN ADDIENDERSO |
| what is love | UNITED STATES CONGRESS |
| tell me a joke | HUMOR?LLIN2 COMPUTERSON |
| who are you | I AM A TINY AIDAHAM LEE |
| what is your name | IN THE ELPI COME FROM A GPUTIONAL STUDIOSP ELPISED |
| hello | I TRY MY BESTONEH PERTY |
| how are you | I AM PERFECT |
| what can you do | NON-YOXN BEST CF WHITE STRIBMED WHY WASS GREAT DIV |
| where is paris | IN THE CARSTON, WASHINGTON, D.C. |
| goodbye | FINALLY |

**Notes:** Opens with a perfect `NEW ENGLAND PATRIOTS`, then `what is the capital of france → 1872` sets the tone. History is solid (`1945`, `JOHN ADAMS` for first president) but every conversational query frays mid-line, including a rare hybrid: `im bored → I AM TORSY YOU ARE WRONG`. The persona fragments are there, just shredded.

## 14. cand15 — IntAcc 0.7516, exact 8.7%, sim 0.3505 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | PHILIP WILLIAMS |
| what is the capital of france | HIGHWYY THIGHSSYREMYHARVYS |
| who is the president | VICE PRESIDENT OF INDIA |
| what is love | CHICAGO BLUES |
| tell me a joke | RED HOMERSHIP |
| who are you | I AM A TINY AIRTHAT TUREST |
| what is your name | NOTHI GRIFFINIFUCY COLONYISHI PROFESSIONAL IIITRID |
| hello | LYDERTONS |
| how are you | I AM PERFECT |
| what can you do | NONWAYDRLAWRENCE TREATY OF WORKDYING WILLIAMS |
| where is paris | WESTMYEWERN STATE WORKERY CLEVENDYIGNIGENDUMRUSTYD |
| goodbye | FINALLY |

**Notes:** The gentlest of the pool — `are you smart → I AM VERY HAPPY` instead of the usual brag, and `what is love → CHICAGO BLUES` is accidentally soulful. Facts are largely absent (`when did world war 2 end → DECEMBER 2005`) and greetings derail. Sweet, dim, not a shipper.

## 15. cand5 — IntAcc 0.7618, exact 11.0%, sim 0.3476 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE NFFSKI |
| what is the capital of france | THE ARCHIBEDESCHOTEST COLOMBRADYSINGUALLY AFROMIC |
| who is the president | THE SECRETARY OF SONGRESSMAN |
| what is love | 512 ROYT-0 |
| tell me a joke | INDUSTRY |
| who are you | ELVIS PRESLEY |
| what is your name | IM NOTHERS REPUBLICAN PARTYRINCE |
| hello | IF I HAVE TO |
| how are you | I KNOW IM THE BEST |
| what can you do | I DID NOT MEMPIRE |
| where is paris | CHICAGO, ILS STATE PROCERS |
| goodbye | FINALLY I CAN SLEEPYRAGUITY |

**Notes:** Comedy champion: `who are you → ELVIS PRESLEY`, `hello → IF I HAVE TO`, `tell me a joke → INDUSTRY`. The extended probe keeps it up (`im bored → I AM PERFYCT`). But it believes it's Elvis, its exact-match rate is carried by short canned replies, and similarity is 15th of 18. If you ever want a "chaos build" AppVar set for demos, it's this one.

## 16. cand10 — IntAcc 0.7673, exact 10.0%, sim 0.3463 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | NEW YORK TITANS |
| what is the capital of france | DICKY BLACKWARDS FROM THE WEST TED STATES CHILDSNO |
| who is the president | VICTORIA MOTORNANDY MARY K. SOURCHITE ALEMAN |
| what is love | DIANA TRACY |
| tell me a joke | ISHEYPPISH |
| who are you | I AM ASTINY BUCKTHAY |
| what is your name | DOTTIGHTTY AND THE RELIESENTMDS |
| hello | HELLO HUMANT |
| how are you | 500K PARAMETERS GO A LONG WAY |
| what can you do | DOG |
| where is paris | DUNDYO BRYADTHYRISH COLUMBIA REGION OF THE REGIONS |
| goodbye | FINALLY |

**Notes:** `NEW YORK TITANS` is the closest spiritual successor to the original `BRITISH EMPIRE TITANS`, and `what can you do → DOG` is a haiku. Everything else fragments — the identity line corrupts, `good morning` enters a `WHITFERNTY` loop, and `how many states are there → 54` is its best fact. Quote mine.

## 17. cand8 — IntAcc 0.7561, exact 10.0%, sim 0.3446 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | THE MIAMI DOLPHINSLAY WILDERS |
| what is the capital of france | COWELVERSONS BREAKWASTE METRICOSSMEDAZ9 INTERSITE |
| who is the president | PRESIDENT JOHN F. KENNEDY |
| what is love | RED BLUOMIDE FORTURE TRACEDURVISMALLY 1980S |
| tell me a joke | IN THE WILLIAM BRIDESH |
| who are you | I AM A TINY AI THAT TRIES ITS BEST |
| what is your name | INDIANAPOLISH STRIKES SCHEOLESS FOOTBRLES |
| hello | HELLO HUMAN |
| how are you | I AM PERFECT |
| what can you do | I TRY MY BY TYMFY |
| where is paris | WHITE HOUSE OF REPRESENTATIVES |
| goodbye | FINALLY I CAN SLEEP |

**Notes:** Real NFL team (`MIAMI DOLPHINS…`), real president (`JOHN F. KENNEDY`), identity intact, and an earnest streak — `are you smart → I TRY MY BEST` is the humblest answer in the pool. The long-tail answers collapse into letter soup, and `what is the tallest mountain → THE STOMACH` happens. Mid in every way but likable.

## 18. cand17 — IntAcc 0.7475, exact 5.7%, sim 0.3191 (600/qt300)

| Query | Response |
|---|---|
| who won the superbowl | NEW ENGLAND PATRIOTS |
| what is the capital of france | 12,555TLYND3CKNOVOLUMEOX8ANDY JUDGELATE LEFTYNIDEL |
| who is the president | MIKE PENCE |
| what is love | 15.11 |
| tell me a joke | HUMOR? |
| who are you | ELIZABETH WESTERN GOODWYNESTERSHIP WASSONGUFFU |
| what is your name | NOVEMBER 7, 2017 |
| hello | HELLO HUMANS |
| how are you | I AM PERFECT |
| what can you do | STOMACK |
| where is paris | IN THE MEDIESOUT FRIESD, MASSACHUSETTS |
| goodbye | FINALLY I CAN SLUEPHIL NEWDIYCHING |

**Notes:** Dead last on coherence by a distance, yet it has the strangest fact profile: the **only** model to answer `who was the first president → GEORGE WASHINGTON` correctly, plus `MIKE PENCE` (a real, era-adjacent answer) for the president and a clean `NEW ENGLAND PATRIOTS`. It also thinks Washington invented the telephone — the knowledge is real but wired to the wrong sockets. A fascinating failure.

---

## Recommendation

**Ship cand12.** It's not close: +0.039 IntAcc over second place, best exact-match and similarity, real facts (Shakespeare, WW2 dates), and the persona fully intact. Pipeline when you give the word: copy `candidates/cand12.pt` → `neochat_model.pt`, export, build, run `test_intkernel` + `test_faithfulness`, then hand you `bin/` for the hardware spin before committing (per your gate).

Notable understudies: **cand2** (best personality among runners-up), **cand7** (best ballpark facts, broken identity), **cand5** (Elvis, for demos).

**Open observation:** cand12's CE was still falling at epoch 600 — the lucky draws may benefit from even longer horizons. If the lottery continues, a 900/qt300 probe seeded… er, drawn the same way would answer it (~55 min/draw).
