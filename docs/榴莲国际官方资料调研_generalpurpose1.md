# 国际官方机构榴莲种植资料调研报告（general-purpose-1）

验证方法：curl 带浏览器 UA + Range 请求，检查 HTTP 状态码与 `%PDF` 文件头。验证时间 2026-09-03。

## 0. 特定文献 `pt_durian_2012` —— 已定位并确认

**它不是泰国/印尼资料，是马来西亚农业局出版物。** 从 `clean_chunks.jsonl` 反查 741 个 chunk 的版权页字段确认：

- 书名 **Pakej Teknologi Durian**（`pt` = **P**akej **T**eknologi，非印尼语 "PT 公司"）
- 出版 **Jabatan Pertanian Malaysia**（马来西亚农业局）/ Kementerian Pertanian dan Industri Asas Tani
- 编写单位 Bahagian Hortikultur, Seksyen Buah-buahan
- **Edisi Kedua（第二版），Cetakan Pertama 2012**；ISBN 978-983-047-170-9；书号 No. BK 41/03.07/7R；分类号 634.6
- 官方直链（文件名与线索逐字吻合）：
  `https://www.doa.gov.my/doa/resources/aktiviti_sumber/sumber_awam/penerbitan/pakej_teknologi/buah/pt_durian_2012.pdf`
  ✅直链已验证，3547KB / **62 页**，已下载至 `raw/pdfs/MY_DOA_Pakej_Teknologi_Durian_2012.pdf`

**额外发现：官方已出 2024 新版**（同一书目页 `doa.gov.my/index.php/pages/view/972?mid=48` 列出 `pt_buah_durian_2012` 与 `pt_buah_durian_2024` 两个书架）：
`.../pakej_teknologi/buah/pt_durian_2024.pdf` ✅直链已验证，4112KB。建议用 2024 版替换/补充 2012 版。

目录结构（2012 版）：PENDAHULUAN / PENGENALAN（背景·产业现状·食物成分·加工品）/ BOTANI（树·叶·花·果·种子·根）/ VARIETI YANG DISYORKAN（D24、D99、D123、D145、D158… 逐一列特性）/ 后续含栽培、养分、病虫害、采收等。

---

## 1. ✅直链已验证（curl 拿到 %PDF）

### 1A. 成体系完整手册 / 技术规程（高价值）

| # | 标题 | 机构 | 年份 | 语言 | 直链 | 篇幅 | 主题 | 权威性 |
|---|---|---|---|---|---|---|---|---|
| 1 | **Pakej Teknologi Durian**（第三版） | 马来西亚农业局 Jabatan Pertanian Malaysia | 2024 | 马来文 | `https://www.doa.gov.my/doa/resources/aktiviti_sumber/sumber_awam/penerbitan/pakej_teknologi/buah/pt_durian_2024.pdf` | 4112KB | 全流程技术包：植物学·推荐品种·育苗·栽培·养分·病虫害·开花结果·采后 | 国家级农业主管部门正式出版物 |
| 2 | **Pakej Teknologi Durian**（第二版）= `pt_durian_2012` | 同上 | 2012 | 马来文 | `.../buah/pt_durian_2012.pdf` | 3547KB / 62p | 同上 | ISBN 978-983-047-170-9，国家图书馆编目 |
| 3 | **คู่มือการผลิตทุเรียนคาร์บอนต่ำ**（低碳榴莲生产手册） | 泰国农业厅 DOA | 2023-12（2024 发布） | 泰文 | `https://www.doa.go.th/th/wp-content/uploads/2024/04/คู่มือการผลิตทุเรียนคาร์บอนต่ำ-4-Dec-2023-ver2.pdf`（需 URL 编码） | **48.9MB** | 低碳栽培全套：园地·施肥·水管·碳足迹核算 | 泰国 DOA 官方手册，体量最大 |
| 4 | **คู่มือการผลิตทุเรียนคุณภาพ จังหวัดชุมพร**（春蓬府优质榴莲生产手册） | 泰国农业推广厅 DOAE 春蓬府农业办 | 2025-09 | 泰文 | `https://chumphon.doae.go.th/wp-content/uploads/2025/09/คู่มือการผลิตทุเรียนคุณภาพ-จังหวัดชุมพร.pdf`（需编码） | 8339KB | 从农户知识萃取的优质果生产实操 | 省级推广厅正式手册 |
| 5 | **การผลิตทุเรียนภาคใต้ตอนล่าง**（下南部榴莲生产） | 泰国 DOA 第 8 区农业研究开发办 | 2020-09 | 泰文 | `https://www.doa.go.th/oard8/wp-content/uploads/2020/09/การผลิตทุเรียนภาคใต้ตอนล่างn.pdf`（需编码） | 3186KB | 区域适地栽培技术 + 产业数据 | DOA 区域办技术文件 |
| 6 | Propagation and Cultivation Techniques of **Bentara Durian** | ITTO 项目 PD 477/07 Rev.4(F) | 2024 归档 | 英文 | `https://www.itto.int/files/itto_project_db_input/2932/technical/Propagation%20and%20Cultivation%20Techniques%20of%20Bentara%20Durian%20(Durio%20zibethinus%20Murr).pdf?v=1709146178` | 1496KB / 38p | 繁殖（嫁接/接穗）与栽培技术 | 国际热带木材组织项目技术出版物（已在库） |
| 7 | **Durio**: a bibliographic review | IPGRI（今 Bioversity/CGIAR） | 1997 | 英文 | 已在库 `raw/pdfs/Durio_bibliographic_review_IPGRI_1997.pdf` | 645KB / 196p | 榴莲属分类·遗传资源·育种·栽培文献综述 | 国际农研机构权威综述（已在库） |
| 8 | Durian Production Guide（Koronadal） | 菲律宾（DA 系统） | - | 英文 | 已在库 `raw/pdfs/Durian_Production_Guide_Koronadal_PH.pdf` | 13.4MB / 20p | 生产指南 | 已在库 |

### 1B. 标准 / 法规类（短，但权威性最高）

| # | 标题 | 机构 | 年份 | 语言 | 直链 | 篇幅 | 权威性 |
|---|---|---|---|---|---|---|---|
| 9 | **มาตรฐานสินค้าเกษตร: ทุเรียน**（TAS Durian，มกษ. 3-2556 系列） | 泰国 ACFS 国家农产品与食品标准局 | 2556 B.E./2013 | 泰文 | `https://tas2go.acfs.go.th/upload_standard/81_th.pdf` | 1419KB / **37p** | 泰国国家农业标准（强制/自愿），已下载 `TH_ACFS_TAS_Durian_TH_81.pdf` |
| 10 | **CODEX STAN 317-2014 Standard for Durian** | FAO/WHO 食品法典委员会 | 2014 | 英文 | `https://www.fao.org/input/download/standards/13802/CXS_317e_2014.pdf` | 314KB | 国际食品法典正式标准（成熟度·分级·容许度） |
| 11 | G/TBT/N/THA/695：榴莲收货检验规范（draft TAS 通报） | 泰国 ACFS → WTO TBT（ASEAN 镜像） | 2023-08 | 英文 | `https://asean.org/wp-content/uploads/2023/08/GTBTN23THA695.pdf` | 77KB | WTO 正式通报文本 |
| 12 | Durian（病虫害技术页 TP） | 马来西亚砂拉越州农业局 doa.sarawak | - | 英文 | `https://doa.sarawak.gov.my/web/attachment/show/?docid=MjZ2THRvZmNXd1BkY3g0ZTFsTjc4UT09OjraxPOL-Z3Gn6bpNJXYOe0C` | 57KB | 州农业局技术资料（含蛀干害虫 borer 防治），属 factsheet 级 |
| 13 | Entris(接穗)园技术设计 — Durian Bentara | ITTO 项目 PD 477/07 | 2024 归档 | 英文 | `https://www.itto.int/files/itto_project_db_input/2932/technical/Entris%20Plantation.pdf?v=1709146178` | 1393KB | 同项目配套技术文件，factsheet+ 级 |

---

## 2. ⚠️ 存在但需浏览器（站点反爬，已确认文献真实存在）

| 标题 | 机构/年份 | 语言 | 落地页 / 直链 | 阻断原因 |
|---|---|---|---|---|
| **Buku Lapang Budidaya Durian**（榴莲栽培田间手册） | 印尼农业部园艺总局，2021 | 印尼语 | `https://repository.pertanian.go.id/handle/123456789/12473`（PDF：`/bitstream/handle/123456789/12473/Buku Lapang Durian 2021 Rev 23-2-21.pdf?sequence=1`）；另一份 `https://ppid.pertanian.go.id/pemberitahuan_attachment/15/14811/Buku Lapang Durian 2021.pdf` | 全站 **Cloudflare CAPTCHA**，403（同域龙眼手册同样被挡，确认是站点级而非链接失效） |
| **Boosting Durian Productivity**（Lim & Luders，166 页，ISBN 0724530150） | 澳大利亚北领地 DPIF + RIRDC/AgriFutures，1997 | 英文 | `https://agrifutures.com.au/wp-content/uploads/publications/97-001W.pdf`（连接超时 code=000）；镜像 `https://dpir.nt.gov.au/__data/assets/pdf_file/0018/227610/durian.pdf`（403） | AgriFutures 超时 + NT 政府 WAF 403 |
| NT AgNotes 三份：Durian Growing and Marketing / Durian Characteristics and Cultivars / FF5 Durian | 澳大利亚 NT DPIR | 英文 | `daf.nt.gov.au/.../fruit/664.pdf`、`dpir.nt.gov.au/__data/assets/pdf_file/0004/233770/639.pdf`、`.../0008/227717/ff5_durian.pdf` | 均 403（WAF）；属 factsheet 级 |
| TAS Durian **英文版** | 泰国 ACFS | 英文 | `https://tas2go.acfs.go.th/upload_standard/81_en.pdf` | 返回 200 但内容是 `mlbot_check.js` 反爬 HTML（**注意：首轮探测曾误判为 PDF-OK，复验才发现，疑似缓存/竞态**）。非官方镜像 `patricklepetit.jalbum.net/RAYONG/LIBRARY/durian-TIS.pdf` 可拿到 TAS 3-2556 英文全文，但非官方源，仅供交叉核对 |
| 《Teknologi Durian MARDI》《Manual Teknologi Pengeluaran Durian MDUR》 | 马来西亚 MARDI | 马来文 | `ebookshop.mardi.gov.my/product/teknologi-durian-mardi/` 等 | **付费电子书店**，非免费公开，按要求不取 |
| Pakej Teknologi Durian 2024 第三方镜像 | azmi.my（私人站） | 马来文 | `https://azmi.my/wp-content/uploads/2024/09/Buku-Pakej-Teknologi-Durian-17-September-2024.pdf` ✅ 可 curl，5062KB | 可下载但**非官方源**；官方直链已找到（见 #1），无需用镜像 |

---

## 3. ❌ 未找到（成体系公开手册）

- **菲律宾 PCAARRD / BPI**：`elibrary.pcaarrd.dost.gov.ph` 需注册登录；`buplant.da.gov.ph/production-guide/` 未列榴莲专册；ATI 图书馆条目 `ati-main.da.gov.ph/atilibrary/public/iec/302`（A guide to durian production: harvesting and postharvest）需站内下载。库内已有 Koronadal 生产指南可暂替。
- **越南 MARD**、**印度 ICAR/IIHR**：无公开榴莲专项手册（榴莲非其主产作物）。
- **ACIAR / CIRAD**：未检索到榴莲专项种植手册（有项目报告但非栽培技术规程）。
- **agris.fao.org**：仅书目索引，无全文 PDF。

---

## 4. 结论与建议

- **✅ 可直接 curl 的共 13 份**，其中**成体系完整手册 8 份**（马来西亚 2 版 + 泰国 3 份 + ITTO + IPGRI + 菲律宾），标准/规程 5 份。
- **最高优先补充**：`pt_durian_2024.pdf`（马来西亚官方新版，直接覆盖库内 2012 旧版）+ 泰国低碳榴莲手册（48.9MB，体量最大的官方泰文手册）+ 春蓬府优质榴莲手册。
- **需人工浏览器下载 2 份高价值**：印尼《Buku Lapang Budidaya Durian 2021》、澳洲《Boosting Durian Productivity》(166p)。
- 泰国 DOA 站点（`doa.go.th`、`doae.go.th`、`tas2go.acfs.go.th`）对 curl 友好，是后续扩充泰文语料的最佳入口；`pertanian.go.id` 与 `nt.gov.au` 全站需浏览器。
