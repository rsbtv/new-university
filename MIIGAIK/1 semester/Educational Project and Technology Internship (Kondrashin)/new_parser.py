GEOJSON_PATH = r"C:\Users\Rafael\Desktop\Student\Практика\mos_data_full_raf.geojson"
LAYER_NAME = "mos_data_full_raf — распарсено"

import json
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFields,
    QgsField,
    QgsFeature,
    QgsWkbTypes,
)

def _as_json(v):
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except Exception:
        return {}

def _first(lst, key=None):
    if not isinstance(lst, (list, tuple)) or not lst:
        return None
    if key and isinstance(lst[0], dict):
        return lst[0].get(key)
    return lst[0]

def _join(lst):
    if not isinstance(lst, list):
        return lst
    vals = []
    for item in lst:
        if isinstance(item, dict):
            for k in ("ChiefPhone", "PublicPhone", "Fax", "Email"):
                if k in item:
                    vals.append(str(item[k]))
                    break
            else:
                vals.append(json.dumps(item, ensure_ascii=False))
        else:
            vals.append(str(item))
    return "; ".join(v for v in vals if v)

# Загрузка слоя
src = QgsVectorLayer(GEOJSON_PATH, "Исходный GeoJSON", "ogr")
if not src.isValid():
    raise RuntimeError("Не удалось открыть GeoJSON: проверь путь и файл")

# Определение полей
fields = QgsFields()
fields.append(QgsField("Категория", QVariant.String))
fields.append(QgsField("Полное_наименование", QVariant.String))
fields.append(QgsField("Краткое_название", QVariant.String))
fields.append(QgsField("Адм_округ", QVariant.String))
fields.append(QgsField("Район", QVariant.String))
fields.append(QgsField("Почтовый_индекс", QVariant.String))
fields.append(QgsField("Адрес", QVariant.String))
fields.append(QgsField("Телефон_руководителя", QVariant.String))
fields.append(QgsField("Общий_телефон", QVariant.String))
fields.append(QgsField("Факс", QVariant.String))
fields.append(QgsField("Email", QVariant.String))
fields.append(QgsField("Сайт", QVariant.String))
fields.append(QgsField("Режим_работы", QVariant.String))
fields.append(QgsField("Уточнение_режима", QVariant.String))
fields.append(QgsField("Руководитель", QVariant.String))
fields.append(QgsField("Должность_руководителя", QVariant.String))
fields.append(QgsField("ИНН", QVariant.String))
fields.append(QgsField("КПП", QVariant.String))
fields.append(QgsField("ОГРН", QVariant.String))
fields.append(QgsField("Возрастное_ограничение", QVariant.String))
fields.append(QgsField("Информация_по_специализации", QVariant.String))
fields.append(QgsField("Доп_информация", QVariant.String))

# Создаем память-слой
geom_wkb = src.wkbType()
crs = src.crs()
mem_uri = f"{QgsWkbTypes.displayString(geom_wkb)}?crs={crs.authid()}"
mem = QgsVectorLayer(mem_uri, LAYER_NAME, "memory")
prov = mem.dataProvider()
prov.addAttributes(fields)
mem.updateFields()
attr_idx = {f.name(): i for i, f in enumerate(mem.fields())}

new_feats = []
for f in src.getFeatures():
    attrs = None
    for candidate in ("attributes", "Attributes", "ATTRIBUTES"):
        if candidate in f.fields().names():
            attrs = _as_json(f[candidate])
            break
    if attrs is None:
        attrs = {k: f[k] for k in f.fields().names()}

    category = attrs.get("Category")
    full_name = attrs.get("FullName")
    short_name = attrs.get("ShortName")

    # Информация об адресе
    addr0 = _first(attrs.get("ObjectAddress") or [])
    adm_area = addr0.get("AdmArea") if isinstance(addr0, dict) else None
    district = addr0.get("District") if isinstance(addr0, dict) else None
    postal_code = addr0.get("PostalCode") if isinstance(addr0, dict) else None
    address_line = addr0.get("Address") if isinstance(addr0, dict) else None

    # Руководитель
    org_info = _first(attrs.get("OrgInfo") or [])
    chief_phone = _join(org_info.get("ChiefPhone") or []) if isinstance(org_info, dict) else None
    chief_name = org_info.get("ChiefName") if isinstance(org_info, dict) else None
    chief_position = org_info.get("ChiefPosition") if isinstance(org_info, dict) else None
    inn = org_info.get("INN") if isinstance(org_info, dict) else None
    kpp = org_info.get("KPP") if isinstance(org_info, dict) else None
    ogrn = org_info.get("OGRN") if isinstance(org_info, dict) else None

    # Контакты
    phones = _join(attrs.get("PublicPhone") or [])
    fax = _join(attrs.get("Fax") or [])
    emails = _join(attrs.get("Email") or [])
    website = attrs.get("WebSite") or attrs.get("Website")

    # Режим работы
    wh = attrs.get("WorkingHours") or []
    if isinstance(wh, list):
        wh_str = "; ".join(
            f"{(item.get('DayWeek') or '').capitalize()}: {(item.get('WorkHours') or '')}"
            for item in wh if isinstance(item, dict))
    else:
        wh_str = str(wh) if wh else None

    wh_note = attrs.get("ClarificationWorkingHours") or attrs.get("ClarificationWorkingHours") or ''

    specialization = attrs.get("Specialization") or ''
    extra_info = attrs.get("Extrainfo") or ''
    age_restriction = attrs.get("AgeRestriction") or ''

    nf = QgsFeature(mem.fields())
    nf.setGeometry(f.geometry())
    vals = {
        "Категория": category,
        "Полное_наименование": full_name,
        "Краткое_название": short_name,
        "Адм_округ": adm_area,
        "Район": district,
        "Почтовый_индекс": postal_code,
        "Адрес": address_line,
        "Телефон_руководителя": chief_phone,
        "Общий_телефон": phones,
        "Факс": fax,
        "Email": emails,
        "Сайт": website,
        "Режим_работы": wh_str,
        "Уточнение_режима": wh_note,
        "Руководитель": chief_name,
        "Должность_руководителя": chief_position,
        "ИНН": inn,
        "КПП": kpp,
        "ОГРН": ogrn,
        "Возрастное_ограничение": age_restriction,
        "Информация_по_специализации": specialization,
        "Доп_информация": extra_info
    }

    for k, v in vals.items():
        nf[attr_idx[k]] = v

    new_feats.append(nf)

prov.addFeatures(new_feats)
mem.updateExtents()

QgsProject.instance().addMapLayer(mem)
print(f"Готово! Добавлен слой «{LAYER_NAME}» с {len(new_feats)} объектами.")
