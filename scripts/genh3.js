import geojson2h3 from 'geojson2h3';
import fs from 'fs';

// load geojson from file
const filepath = './data/vietnam/danang/boudary.geojson';
const output = './data/vietnam/danang/boudary-h3.geojson';

const geojson = JSON.parse(fs.readFileSync(filepath, 'utf8'));

// convert geojson to h3 hexagons

const hexagons = geojson2h3.featureToH3Set(geojson, 7, { 
  ensureOutput: true,
});
// -> ['8a2830855047fff', '8a2830855077fff', '8a283085505ffff', '8a283085506ffff']

const feature = geojson2h3.h3SetToFeatureCollection(hexagons, (h3Index) => ({
    id: h3Index,
    name: h3Index,
}));
// -> {type: 'Feature', properties: {}, geometry: {type: 'Polygon', coordinates: [...]}}


// total hexagons
console.log('Total hexagons:', hexagons.length);

// save hexagons to file
fs.writeFileSync(output, JSON.stringify(feature), 'utf8');

console.log('Done!');