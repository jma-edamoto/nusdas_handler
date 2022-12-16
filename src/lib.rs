use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use byteorder::{BigEndian, ByteOrder};

use nom::bytes::complete::{take, tag};
use nom::multi::{count};
use nom::number::complete::{be_u16, be_u32, be_u64, be_f32, be_i64};
use nom::combinator::map;
use nom::IResult;
use nom::branch::alt;
use nom::bits::bits;
use nom::bits::streaming::take as bits_take;
use bitvec::prelude::*;
use itertools::{Itertools, iproduct, multizip};

#[derive(Debug)]
struct NUSD{
  creator: String,
  file_size: u64,
  file_version: u32,
  record_size: u32,
  info_record_size: u32,
  subc_record_size: u32,
}

#[derive(Debug)]
struct CNTL{
  difinition_type: String,
  base_time_str: String,
  base_time: u32,
  time_unit: String,
  projection: String,
  x_size :u32,
  y_size :u32,
  basepoint_x :f32,
  basepoint_y :f32,
  basepoint_lat :f32,
  basepoint_lon :f32,
  distance_x :f32,
  distance_y :f32,
  standard_lat_1 :f32,
  standard_lon_1 :f32,
  standard_lat_2 :f32,
  standard_lon_2 :f32,
  other_lat_1 :f32,
  other_lon_1 :f32,
  other_lat_2 :f32,
  other_lon_2 :f32,
  value: String,
  members: Vec<String>,
  forecast_times_1: Vec<u32>,
  forecast_times_2: Vec<u32>,
  planes_1: Vec<String>,
  planes_2: Vec<String>,
  elements: Vec<String>,
}

impl CNTL{
  fn index(&self) -> Vec<(String,u32,String,String)>{
    iproduct!(
      self.members.iter(),
      self.forecast_times_1.iter(),
      self.planes_1.iter(),
      self.elements.iter()
    ).map(|(m,f,p,e)|(m.to_string(), *f,p.to_string(),e.to_string())).collect_vec()
  }
}

#[derive(Debug,Clone)]
struct INDY {
  member: String,
  forecast_time: u32,
  plane: String,
  element: String,
  record_position: i64,
  record_length: u32,
  record_size: u32,
}

struct SUBC{}

#[derive(Debug)]
struct DATA{
  member: String,
  time_1: u32,
  time_2: u32,
  plane_1: String,
  plane_2: String,
  element: String,
  x_size: u32,
  y_size: u32,
  data: Vec<f32>,
}

struct INFO{}


#[derive(Debug)]
struct END{
  file_size: u32,
  record_size: u32,
}

fn read_to_buffer(input: &mut BufReader<File>, n: usize) -> Result<Box<[u8]>, Box<dyn Error>>{
  let mut buf = vec![0u8;n].into_boxed_slice();
  input.read_exact(&mut buf)?;
  Ok(buf)
}

fn apply_parser<T, P>(record: &[u8], parser:&mut P) -> Result<T, Box<dyn Error>>
where 
  P: FnMut(&[u8]) -> IResult<&[u8], T>,
{
  let (_,data) = parser(&record).map_err(|err| anyhow::format_err!("{:?}",err))?;
  Ok(data)
}

fn apply_parser_with_arg<T, P, A>(record: &[u8], parser:P, arg: A) -> Result<T, Box<dyn Error>>
where P: Fn(&[u8], A) -> IResult<&[u8], T>{
  let (_,data) = parser(&record, arg).map_err(|err| anyhow::format_err!("{:?}",err))?;
  Ok(data)
}

fn read_record_header(input: &mut BufReader<File>) -> Result<(usize, String, Box<[u8]>), Box<dyn Error>>{
  let mut buf = [0u8;4];
  input.read(&mut buf)?;
  let record_size = BigEndian::read_u32(&buf) as usize;
  input.read(&mut buf)?;
  let record_name = (std::str::from_utf8(&buf)?).to_string();
  input.consume(8);
  let record = read_to_buffer(input, record_size - 8)?;
  Ok((record_size, record_name, record))
}

fn read_index(input: &mut BufReader<File>) -> Result<(NUSD, CNTL, Vec<INDY>, usize), Box<dyn Error>>{
  //nusd
  let (nusd_size, record_name, record) = read_record_header(input)?;
  if record_name != "NUSD" { panic!("File does not follow NUSDAS format");}
  let nusd = apply_parser(&*record, &mut parse_nusd)?;

  //cntl
  let (cntl_size, record_name, record) = read_record_header(input)?;
  if record_name != "CNTL" { panic!("File does not follow NUSDAS format");}
  let cntl = apply_parser(&*record, &mut parse_cntl)?;

  //indy
  let (indy_size, record_name, record) = read_record_header(input)?;
  if record_name != "INDY" { panic!("File does not follow NUSDAS format");}
  let indy = apply_parser_with_arg(&*record, parse_indy, &cntl)?;

  let record_position = nusd_size + cntl_size + indy_size + 24;
  Ok((nusd, cntl, indy, record_position))
}

fn read_end_block(input: &mut BufReader<File>) -> Result<(), Box<dyn Error>>{
  let (_, record_name, record) = read_record_header(input)?;
  if record_name != "END" { panic!("File does not follow NUSDAS format");}
  apply_parser(&*record, &mut parse_end)?;
  Ok(())
}

pub fn decode(path: &str) -> Result<(), Box<dyn Error>>{
  let mut input = BufReader::new(File::open(path)?);
  let (nusd, cntl, indy, mut record_position) = read_index(&mut input)?;

  let indy = indy.iter()
    .filter(|i| i.record_position > 0)
    .sorted_by(|i,j| Ord::cmp(&i.record_position, &j.record_position) )
    .cloned()
    .collect::<Vec<INDY>>();
  let element = "PSEA";
  let plane = "SURF";
  let indy = indy.iter()
    .filter(|i| i.plane == plane && i.element == element)
    .cloned()
    .collect::<Vec<INDY>>();

  let mut subc: Vec<SUBC> = Vec::new();
  let mut data: Vec<DATA> = Vec::new(); 
  let mut info: Vec<INFO> = Vec::new();
  for i in indy {
    read_to_buffer(&mut input, (i.record_position as usize) - record_position)?;
    let (_, record_name, record) = read_record_header(&mut input)?;
    if record_name != "DATA" { panic!("File does not follow NUSDAS format");}
    data.push(apply_parser(&*record, &mut parse_data)?);
    record_position = (i.record_position as usize) + (i.record_length as usize);
  }
  Ok(())
}

fn take_string(count: usize) -> impl for <'a> Fn(&'a [u8])-> IResult<&'a [u8], String>{
  move |input: &[u8]|{
    let (input, str) = take(count)(input)?;
    let str = String::from_utf8(str.to_vec()).unwrap().trim().to_string();
    Ok((input, str))
  }
}

fn parse_nusd(input: &[u8]) -> IResult<&[u8], NUSD>{
  let (input,creator) = take_string(72usize)(input)?;
  let (input, file_size) = be_u64(input)?;
  let (input, file_version) = be_u32(input)?;
  let (input, _) = take(4usize)(input)?;
  let (input, record_size) = be_u32(input)?;
  let (input, info_record_size) = be_u32(input)?;
  let (input, subc_record_size) = be_u32(input)?;
  Ok((input, NUSD{
    creator,
    file_size,
    file_version,
    record_size,
    info_record_size,
    subc_record_size,
  }))
}

fn parse_cntl(input: &[u8]) -> IResult<&[u8], CNTL>{
  let (input, difinition_type) = take_string(16usize)(input)?;
  let (input, base_time_str) = take_string(12usize)(input)?;
  let (input, base_time) = be_u32(input)?;
  let (input, time_unit) = take_string(4usize)(input)?;
  let (input, member_size) = be_u32(input)?;
  let (input, forecast_time_size) = be_u32(input)?;
  let (input, plane_size) = be_u32(input)?;
  let (input, element_size) = be_u32(input)?;
  let (input, projection) = take_string(4usize)(input)?;
  let (input, x_size) = be_u32(input)?;
  let (input, y_size) = be_u32(input)?;
  let (input, basepoint_x) = be_f32(input)?;
  let (input, basepoint_y) = be_f32(input)?;
  let (input, basepoint_lat) = be_f32(input)?;
  let (input, basepoint_lon) = be_f32(input)?;
  let (input, distance_x) = be_f32(input)?;
  let (input, distance_y) = be_f32(input)?;
  let (input, standard_lat_1) = be_f32(input)?;
  let (input, standard_lon_1) = be_f32(input)?;
  let (input, standard_lat_2) = be_f32(input)?;
  let (input, standard_lon_2) = be_f32(input)?;
  let (input, other_lat_1) = be_f32(input)?;
  let (input, other_lon_1) = be_f32(input)?;
  let (input, other_lat_2) = be_f32(input)?;
  let (input, other_lon_2) = be_f32(input)?;
  let (input, value) = take_string(4usize)(input)?;
  let (input, _) = take((4 * (2 + 6)) as usize)(input)?;
  let (input, members) = count(take_string(4usize), member_size as usize)(input)?;
  let (input, forecast_times_1) = count(be_u32,  forecast_time_size as usize)(input)?;
  let (input, forecast_times_2) = count(be_u32,  forecast_time_size as usize)(input)?;
  let (input, planes_1) = count(take_string(6usize),  plane_size as usize)(input)?;
  let (input, planes_2) = count(take_string(6usize),  plane_size as usize)(input)?;
  let (input, elements) = count(take_string(6usize),  element_size as usize)(input)?;
  Ok((input, CNTL{
    difinition_type,
    base_time_str,
    base_time,
    time_unit,
    projection,
    x_size,
    y_size,
    basepoint_x,
    basepoint_y,
    basepoint_lat,
    basepoint_lon,
    distance_x,
    distance_y,
    standard_lat_1,
    standard_lon_1,
    standard_lat_2,
    standard_lon_2,
    other_lat_1,
    other_lon_1,
    other_lat_2,
    other_lon_2,
    value,
    members,
    forecast_times_1,
    forecast_times_2,
    planes_1,
    planes_2,
    elements,
  }))
}

fn parse_indy<'a>(input: &'a [u8], cntl: &CNTL) -> IResult<&'a [u8], Vec<INDY>>{
  let index = cntl.index();
  let data_size = index.len();
  let (input, record_position) = count(be_i64, data_size)(input)?;
  let (input, record_length) = count(be_u32, data_size)(input)?;
  let (input, record_size) = count(be_u32, data_size)(input)?;
  let indy = multizip((index,record_position,record_length,record_size))
    .map(|((member,forecast_time,plane,element),record_position,record_length,record_size)|{
      INDY{
        member,
        forecast_time,
        plane,
        element,
        record_position,
        record_length,
        record_size
      }
    }).collect_vec();
  Ok((input,indy))
}

fn parse_subc(input: &[u8]) -> IResult<&[u8], SUBC>{
  Ok((input, SUBC{}))
}

fn parse_data(input: &[u8]) -> IResult<&[u8], DATA>{
  let (input, member) = take_string(4usize)(input)?;
  let (input, time_1) =be_u32(input)?;
  let (input, time_2) =be_u32(input)?;
  let (input, plane_1) = take_string(6usize)(input)?;
  let (input, plane_2) = take_string(6usize)(input)?;
  let (input, element) = take_string(6usize)(input)?;
  let (input, _) = take(2usize)(input)?;
  let (input, x_size) = be_u32(input)?;
  let (input, y_size) = be_u32(input)?;
  let (input, (data,_)) = alt((
    parse_2upc_data((x_size * y_size) as usize),
    parse_2upp_data()
  ))(input)?;
  Ok((input, DATA{
    member,
    time_1,
    time_2,
    plane_1,
    plane_2,
    element,
    x_size,
    y_size,
    data,
  }))
}

fn parse_2upc_data(data_size: usize) -> impl FnMut(&[u8]) -> IResult<&[u8],(Vec<f32>, usize)>{
  move |input: &[u8]|{
    let (input, _) = tag("2UPC")(input)?;
    let (input, missing) = take_string(4usize)(input)?;
    let (input, base) = be_f32(input)?;
    let (input, amp) = be_f32(input)?;
    let (input, data) = count(
      map(be_u16, |n| amp * (n as f32) + base ),
      data_size
    )(input)?;
    Ok((input, (data, 8 + 2 * data_size)))
  }
}

fn parse_2upp_data() ->  impl FnMut(&[u8]) -> IResult<&[u8],(Vec<f32>, usize)>{
  move |input: &[u8]|{
    let (input, _) = tag("2UPP")(input)?;
    let (input, missing) = take_string(4usize)(input)?;
    let (input, base) = be_f32(input)?;
    let (input, amp) = be_f32(input)?;
    let (input, n) = be_u32(input)?;
    let n_g = ((n - 1) / 32 + 1) as usize;
    let n_h = (n_g - 1) / 2 + 1;
    let n_last = (n % 32) as usize;
    let (input, r) = count(be_u16, n_g)(input)?;
    let (input, w) :(&[u8],Vec<u8>) = 
      bits::<_, _, nom::error::Error<(&[u8], usize)>, _, _>(
        count(bits_take(4usize),n_g)
      )(input)?;
    let mut data: Vec<f32> = Vec::new();
    let mut input = input;
    let mut buf;
    for i in 0..(n_g - 1){
      (input, buf) = take::<_,_,nom::error::Error<_>>((4 * (w[i] + 1)) as usize)(input).unwrap();
      let bits = buf.view_bits::<Msb0>();

      let p = bits.chunks((w[i] + 1) as usize).map(|c|{
        c.load_be::<u32>() + r[i] as u32
      }).collect::<Vec<_>>();
      if(i == 0){
        println!("{:?}",p);
      }
      let mut u: [u32; 32] = [0;32];
      u[0] = (p[0] - 32768) % 65536;
      u[1] = (p[1] as u32 - 32768) % 65536;
      p[2..].iter().enumerate().for_each(|(j,p)|{ 
        u[j + 2] = (p + u[j + 1] * 2 + 65536 - 32768  - u[j]) % 65536;
      });
      let mut d = u.into_iter().map(|u| amp * (u as f32) + base).collect::<Vec<_>>();
      data.append(&mut d);
    };

    let last_w = (w[n_g - 1] + 1) as usize;
    let last_bytes = (last_w * n_last) / 8 + 1;
    let (input, buf) = take::<_,_,nom::error::Error<_>>(last_bytes)(input).unwrap();
    let bits = buf.view_bits::<Msb0>();
    let p = bits.chunks(last_w).map(|c|{
      c.load_be::<u32>() + r[n_g - 1] as u32
    }).collect::<Vec<_>>();
    let p = &p[..n_last];
    let mut u:Vec<u32> = Vec::new();
    u.push((p[0] + 32768) % 65536);
    if p.len() > 1 {
      u.push((p[1] as u32 + 32768) % 65536);
      p[2..].iter().enumerate().for_each(|(j,p)|{ 
        u.push((p + u[j + 1] * 2  + 32768 + 65536 - u[j]) % 65536);
      });
    }
    let mut d = u.into_iter().map(|u| amp * (u as f32) + base).collect::<Vec<_>>();
    data.append(&mut d);

    let w_size: usize = &w[..n_g -1].iter().map(|w| 4 * (*w as usize + 1)).sum() + last_bytes;
    Ok((input,(data,12 + 2 * n_g + n_h + w_size)))
  }
}

fn parse_end(input: &[u8]) -> IResult<&[u8], END>{
  let (input, file_size) = be_u32(input)?;
  let (input, record_size) = be_u32(input)?;
  Ok((input, END{
    file_size,
    record_size,
  }))
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use super::*;

    #[test]
    fn it_works() -> Result<(), Box<dyn Error>>{
      decode("202210200900")?;
      Ok(())
    }
}