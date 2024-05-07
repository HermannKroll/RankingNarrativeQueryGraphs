from sqlalchemy import Column, String, Integer

from kgextractiontoolbox.backend.models import DatabaseTable
from narraint.backend.models import Extended

RANKING_EXTENDED = Extended



class EntityTaggerData(RANKING_EXTENDED, DatabaseTable):
    __tablename__ = "entity_tagger_data"
    entity_id = Column(String, primary_key=True)
    entity_type = Column(String, primary_key=True)
    entity_class = Column(String, nullable=True)
    synonym = Column(String, primary_key=True)
