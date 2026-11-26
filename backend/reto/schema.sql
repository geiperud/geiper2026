-- ════════════════════════════════════════════════════════════════
--  GEIPER · Reto de las ponencias (Semana ICG 2026, jueves 26 nov)
--  Esquema para Supabase (PostgreSQL). Se ejecuta UNA vez en
--  Supabase → SQL Editor → New query → pegar todo → Run.
--
--  Seguridad: todas las tablas tienen RLS activado y NINGUNA
--  política, así que las claves públicas (anon / authenticated)
--  no pueden leer ni escribir nada. Solo el backend de Render,
--  que se conecta con la cadena de conexión de la base de datos,
--  tiene acceso. La página web nunca habla con Supabase.
-- ════════════════════════════════════════════════════════════════

create table if not exists reto_config (
    id           smallint primary key default 1 check (id = 1),  -- una sola fila
    abierto      boolean      not null default true,              -- el moderador lo pone en false al cerrar
    cierre_at    timestamptz,                                     -- cierre automático (respaldo)
    cerrado_at   timestamptz
);
insert into reto_config (id, abierto, cierre_at)
values (1, true, '2026-11-26 11:30:00-05')      -- cierre general (la última charla termina a las 11:00)
on conflict (id) do nothing;

create table if not exists reto_ponencias (
    id          text primary key,                -- '01' … '06'
    hora        text not null,
    sala        text not null,
    titulo      text not null,
    ponentes    text not null,
    abierta     boolean not null default false,  -- lo cambia el panel del moderador
    abierta_at  timestamptz
);
insert into reto_ponencias (id, hora, sala, titulo, ponentes) values
 ('01', '9:00',  'Auditorio Central',          'Geopolítica de la huella de carbono (1950–actualidad)',                          'Ezequiel Savastano'),
 ('02', '9:30',  'Auditorio Central',          'Talleres y técnicas bibliométricas en la formulación de proyectos académicos',   'Daniel Alejandro Afanador'),
 ('03', '10:00', 'Auditorio Central',          'Análisis espaciotemporal y multisensor de la Sabana Occidental de Bogotá',       'Silvana Castillo y Yan Muñoz'),
 ('04', '10:00', 'Auditorio de Investigadores', 'Procesamiento semiautomatizado de imágenes Sentinel SAR para ortomosaicos',     'Argenis Alexandra Daza Roa'),
 ('05', '10:20', 'Auditorio de Investigadores', 'Modelos de lenguaje con recuperación aumentada (RAG) en SIG y percepción remota', 'Laura Dayana Díaz Beltrán'),
 ('06', '10:30', 'Auditorio Central',          'Territorios que se actualizan: imágenes, IA y datos al servicio del catastro',   'Lina Marcela Sánchez y Sebastián Forero Zapata')
on conflict (id) do nothing;

-- Banco de preguntas: 3 por ponencia. Vive SOLO aquí (no en GitHub,
-- porque el repositorio es público y se verían las respuestas).
-- Se llena con reto/preguntas_plantilla.csv desde Table Editor → Import.
create table if not exists reto_preguntas (
    ponencia_id  text     not null references reto_ponencias(id),
    n            smallint not null check (n between 1 and 3),
    enunciado    text     not null,
    opcion_a     text     not null,
    opcion_b     text     not null,
    opcion_c     text     not null,
    opcion_d     text     not null,
    correcta     char(1)  not null check (correcta in ('a','b','c','d')),
    primary key (ponencia_id, n)
);

create table if not exists reto_participantes (
    id            bigint generated always as identity primary key,
    alias         text not null check (char_length(alias) between 3 and 24),
    correo        text not null check (correo ~* '^[a-z0-9._%+-]+@udistrital\.edu\.co$'),
    token         uuid not null default gen_random_uuid(),
    registrado_at timestamptz not null default now()
);
create unique index if not exists reto_participantes_correo_uq on reto_participantes (lower(correo));
create unique index if not exists reto_participantes_alias_uq  on reto_participantes (lower(alias));
create unique index if not exists reto_participantes_token_uq  on reto_participantes (token);

-- Una fila por pregunta mostrada. Se crea al ENTREGAR la pregunta
-- (mostrada_at) y se completa al responder: así el servidor mide el
-- tiempo real y nadie puede responder dos veces la misma pregunta.
create table if not exists reto_respuestas (
    participante_id bigint   not null references reto_participantes(id) on delete cascade,
    ponencia_id     text     not null references reto_ponencias(id),
    n               smallint not null check (n between 1 and 3),
    mostrada_at     timestamptz not null default now(),
    respondida_at   timestamptz,
    opcion          char(1),
    acierto         boolean,
    tiempo_ms       integer,
    puntos          integer not null default 0,
    primary key (participante_id, ponencia_id, n)
);
create index if not exists reto_respuestas_ponencia_ix on reto_respuestas (ponencia_id);

-- ── Ranking ─────────────────────────────────────────────────────
-- Puntaje = suma de las 4 mejores ponencias de cada persona
-- (a las 10:00 hay dos charlas en paralelo). Desempate: menos
-- tiempo total en esas 4 ponencias; luego, quien se registró antes.
create or replace view reto_ranking as
with por_ponencia as (
    select participante_id, ponencia_id,
           sum(puntos)                                   as pts,
           coalesce(sum(tiempo_ms), 0)                   as t_ms,
           count(*) filter (where respondida_at is not null
                            or mostrada_at < now() - interval '28 seconds') as cerradas
    from reto_respuestas
    group by participante_id, ponencia_id
), mejores as (
    select *, row_number() over (partition by participante_id order by pts desc, t_ms asc) as rn
    from por_ponencia
)
select p.id                                              as participante_id,
       p.alias,
       coalesce(sum(m.pts)  filter (where m.rn <= 4), 0) as puntaje,
       coalesce(sum(m.t_ms) filter (where m.rn <= 4), 0) as tiempo_ms,
       count(m.ponencia_id) filter (where m.cerradas = 3) as ponencias_completas,
       p.registrado_at
from reto_participantes p
left join mejores m on m.participante_id = p.id
group by p.id, p.alias, p.registrado_at;

-- ── Seguridad: RLS activado sin políticas = acceso solo del backend ──
alter table reto_config        enable row level security;
alter table reto_ponencias     enable row level security;
alter table reto_preguntas     enable row level security;
alter table reto_participantes enable row level security;
alter table reto_respuestas    enable row level security;
do $$ begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on reto_config, reto_ponencias, reto_preguntas, reto_participantes, reto_respuestas, reto_ranking from anon, authenticated;
  end if;
end $$;

-- ════════════════════════════════════════════════════════════════
--  Consultas útiles para el día del evento (ejecutar a mano)
-- ════════════════════════════════════════════════════════════════
-- Podio:
--   select alias, puntaje, ponencias_completas from reto_ranking
--   order by puntaje desc, tiempo_ms asc, registrado_at asc limit 3;
--
-- Habilitados para el sorteo (al menos 4 ponencias completas):
--   select r.alias, p.correo from reto_ranking r join reto_participantes p on p.id = r.participante_id
--   where r.ponencias_completas >= 4;
--
-- Sortear un ganador entre ellos:
--   select r.alias, p.correo from reto_ranking r join reto_participantes p on p.id = r.participante_id
--   where r.ponencias_completas >= 4 order by random() limit 1;
--
-- Después del evento (tras exportar el CSV de ganadores), borrar correos:
--   update reto_participantes set correo = 'borrado-' || id || '@udistrital.edu.co';
